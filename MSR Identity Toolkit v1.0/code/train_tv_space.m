function T = train_tv_space(dataList, ubmFilename, tv_dim, niter, nworkers, tvFilename)
% uses statistics in dataLits to train the i-vector extractor with tv_dim 
% factors and niter EM iterations. The training process can be parallelized
% via parfor with nworkers. The output can be optionally saved in tvFilename.
%
% Technically, assuming a factor analysis (FA) model of the from:
%
%           M = m + T . x
%
% for mean supervectors M, the code computes the maximum likelihood 
% estimate (MLE)of the factor loading matrix T (aka the total variability 
% subspace). Here, M is the adapted mean supervector, m is the UBM mean 
% supervector, and x~N(0,I) is a vector of total factors (aka i-vector).
%
% Inputs:
%   - dataList    : ASCII file containing stats file names (1 file per line)
%                   or a cell array of concatenated stats (i.e., [N; F])
%   - ubmFilename : UBM file name or a structure with UBM hyperparameters
%   - tv_dim      : dimensionality of the total variability subspace
%   - niter       : number of EM iterations for total subspace learning
%   - nworkers    : number of parallel workers 
%   - tvFilename  : output total variability matrix file name (optional)
%
% Outputs:
%   - T 		  : total variability subspace matrix  
%
% References:
%   [1] D. Matrouf, N. Scheffer, B. Fauve, J.-F. Bonastre, "A straightforward 
%       and efficient implementation of the factor analysis model for speaker 
%       verification," in Proc. INTERSPEECH, Antwerp, Belgium, Aug. 2007, 
%       pp. 1242-1245.  
%   [2] P. Kenny, "A small footprint i-vector extractor," in Proc. Odyssey, 
%       The Speaker and Language Recognition Workshop, Singapore, Jun. 2012.
%   [3] N. Dehak, P. Kenny, R. Dehak, P. Dumouchel, and P. Ouellet, "Front-end 
%       factor analysis for speaker verification," IEEE TASLP, vol. 19, pp. 
%       788-798, May 2011. 
%
%
% Omid Sadjadi <s.omid.sadjadi@gmail.com>
% Microsoft Research, Conversational Systems Research Center

if ischar(tv_dim), tv_dim = str2double(tv_dim); end
if ischar(niter), niter = str2double(niter); end
if ischar(nworkers), nworkers = str2double(nworkers); end

if ischar(ubmFilename),
	tmp  = load(ubmFilename);
	ubm  = tmp.gmm;
elseif isstruct(ubmFilename),
	ubm = ubmFilename;
else
    error('Oops! ubmFilename should be either a string or a structure!');
end
[ndim, nmix] = size(ubm.mu);
S = reshape(ubm.sigma, ndim * nmix, 1);

[N, F] = load_data(dataList, ndim, nmix);
if iscell(dataList), clear dataList; end

fprintf('\n\nRandomly initializing T matrix ...\n\n');
% suggested in jfa cookbook
T = randn(tv_dim, ndim * nmix) * sum(S) * 0.001;

fprintf('Re-estimating the total subspace with %d factors ...\n', tv_dim);
for iter = 1 : niter,
    fprintf('EM iter#: %d \t', iter);
    tim = tic;
    [LU, RU] = expectation_tv(T, N, F, S, tv_dim, nmix, ndim, nworkers);
    T = maximization_tv(LU, RU, ndim, nmix);
    tim = toc(tim);
    fprintf('[elaps = %.2f s]\n', tim);
end

if ( nargin == 6 ),
	fprintf('\nSaving T matrix to file %s\n', tvFilename);
	% create the path if it does not exist and save the file
	path = fileparts(tvFilename);
	if ( exist(path, 'dir')~=7 && ~isempty(path) ), mkdir(path); end
	save(tvFilename, 'T');
end

function [N, F] = load_data(datalist, ndim, nmix)
% load all data into memory
if ischar(datalist),
    fid = fopen(datalist, 'rt');
    filenames = textscan(fid, '%s');
    fclose(fid);
    filenames = filenames{1};
    nfiles = size(filenames, 1);
    N = zeros(nfiles, nmix, 'single');
    F = zeros(nfiles, ndim * nmix, 'single');
    for file = 1 : nfiles,
        tmp = load(filenames{file});
        N(file, :) = tmp.N;
        F(file, :) = tmp.F;
    end
else
    nfiles = length(datalist);
    N = zeros(nfiles, nmix, 'single');
    F = zeros(nfiles, ndim * nmix, 'single');
    for file = 1 : nfiles,
        N(file, :) = datalist{file}(1:nmix);
        F(file, :) = datalist{file}(nmix + 1 : end);
    end
end



function [LU, RU] = expectation_tv(T, N, F, S, tv_dim, nmix, ndim, nworkers)
% compute the posterior means and covariance matrices of the factors 
% or latent variables
idx_sv = reshape(repmat(1 : nmix, ndim, 1), ndim * nmix, 1);
nfiles = size(F, 1);

LU = cell(nmix, 1);
LU(:) = {zeros(tv_dim)};

RU = zeros(tv_dim, nmix * ndim);
I = eye(tv_dim);
T_invS =  bsxfun(@rdivide, T, S');

parts = 250; % modify this based on your resources
nbatch = floor( nfiles/parts + 0.99999 );
for batch = 1 : nbatch,
    start = 1 + ( batch - 1 ) * parts;
    fin = min(batch * parts, nfiles);
    len = fin - start + 1;
    index = start : fin;
    N1 = N(index, :);
    F1 = F(index, :);
    Ex = zeros(tv_dim, len);
    Exx = zeros(tv_dim, tv_dim, len);
    parfor (ix = 1 : len, nworkers)
%     for ix = 1 : len,
        L = I +  bsxfun(@times, T_invS, N1(ix, idx_sv)) * T';
        Cxx = pinv(L); % this is the posterior covariance Cov(x,x)
        B = T_invS * F1(ix, :)';
        Ex(:, ix) = Cxx * B; % this is the posterior mean E[x]
        Exx(:, :, ix) = Cxx + Ex(:, ix) * Ex(:, ix)';
    end
    RU = RU + Ex * F1;
    parfor (mix = 1 : nmix, nworkers)
%     for mix = 1 : nmix,
        tmp = bsxfun(@times, Exx, reshape(N1(:, mix),[1 1 len]));
        LU{mix} = LU{mix} + sum(tmp, 3);
    end
end

function RU = maximization_tv(LU, RU, ndim, nmix)
% ML re-estimation of the total subspace matrix or the factor loading
% matrix
for mix = 1 : nmix
    idx = ( mix - 1 ) * ndim + 1 : mix * ndim;
    RU(:, idx) = LU{mix} \ RU(:, idx);
end
