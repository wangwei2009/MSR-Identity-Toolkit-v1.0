function gmm = mapAdapt(dataList, ubmFilename, tau, config, gmmFilename)
% MAP-adapts a speaker specific GMM gmmFilename from UBM ubmFilename using
% features in dataList. The MAP relevance factor can be specified via tau.
% Adaptation of all GMM hyperparameters are supported. 
%
% Inputs:
%   - dataList    : ASCII file containing adaptation feature file name(s) 
%                   or a cell array containing feature(s). Feature files 
%					must be in uncompressed HTK format.  
%   - ubmFilename : file name of the UBM or a structure containing 
%					the UBM hyperparameters that is,
%					(ubm.mu: means, ubm.sigma: covariances, ubm.w: weights)
%   - tau         : the MAP adaptation relevance factor (19.0)
%   - config      : any sensible combination of 'm', 'v', 'w' to adapt 
%                   mixture means (default), covariances, and weights
%   - gmmFilename : the output speaker specific GMM file name (optional)
%
% Outputs:
%   - gmm		  : a structure containing the GMM hyperparameters
%					(gmm.mu: means, gmm.sigma: covariances, gmm.w: weights)
%
% References:
%   [1] D.A. Reynolds, T.F. Quatieri, and R.B. Dunn, "Speaker verification 
%       using adapted Gaussian mixture models," Digital Signal Process., 
%       vol. 10, pp. 19-41, Jan. 2000.
%
%
% Omid Sadjadi <s.omid.sadjadi@gmail.com>
% Microsoft Research, Conversational Systems Research Center

if ( nargin < 3 ), 
    tau = 19.0; % MAP adaptation relevance factor
end
if ( nargin < 4 ), config = ''; end;

if ischar(tau), tau = str2double(tau); end

if isempty(config), config = 'm'; end

if ischar(ubmFilename),
	tmp = load(ubmFilename);
	ubm = tmp.gmm;
elseif isstruct(ubmFilename),
	ubm = ubmFilename;
else
	error('oh dear! ubmFilename should be either a string or a structure!');
end

gmm = ubm;

if ischar(dataList) || iscellstr(dataList),
	dataList = load_data(dataList);
end
if ~iscell(dataList),
	error('Oops! dataList should be a cell array!');
end
nfiles = length(dataList);

N = 0; F = 0; S = 0;
parfor file = 1 : nfiles,
    [n, f, s] = expectation(dataList{file}, ubm);
    N = N + n; F = F + f; S = S + s;
end

if any(config == 'm'),
	alpha = N ./ (N + tau); % tarde-off between ML mean and UBM mean
	m_ML = bsxfun(@rdivide, F, N);
	m = bsxfun(@times, ubm.mu, (1 - alpha)) + bsxfun(@times, m_ML, alpha); 
	gmm.mu = m;
end

if any(config == 'v'),
	alpha = N ./ (N + tau);
	v_ML = bsxfun(@rdivide, S, N);
	v = bsxfun(@times, (ubm.sigma+ubm.mu.^2), (1 - alpha)) + bsxfun(@times, v_ML, alpha) - (m .* m); 
	gmm.sigma = v;
end

if any(config == 'w'),
	alpha = N ./ (N + tau);
	w_ML = N / sum(N);
	w = bsxfun(@times, ubm.w, (1 - alpha)) + bsxfun(@times, w_ML, alpha); 
	w = w / sum(w);
	gmm.w = w;
end

if ( nargin == 5 ),
	% create the path if it does not exist and save the file
	path = fileparts(gmmFilename);
	if ( exist(path, 'dir')~=7 && ~isempty(path) ), mkdir(path); end
	save(gmmFilename, 'gmm');
end

function data = load_data(datalist)
% load all data into memory
if ~iscellstr(datalist)
    fid = fopen(datalist, 'rt');
    filenames = textscan(fid, '%s');
    fclose(fid);
    filenames = filenames{1};
else
    filenames = datalist;
end
nfiles = size(filenames, 1);
data = cell(nfiles, 1);
for ix = 1 : nfiles,
    data{ix} = htkread(filenames{ix});
end

function [N, F, S, llk] = expectation(data, gmm)
% compute the sufficient statistics
[post, llk] = postprob(data, gmm.mu, gmm.sigma, gmm.w(:));
N = sum(post, 2)';
F = data * post';
S = (data .* data) * post';

function [post, llk] = postprob(data, mu, sigma, w)
% compute the posterior probability of mixtures for each frame
post = lgmmprob(data, mu, sigma, w);
llk  = logsumexp(post, 1);
post = exp(bsxfun(@minus, post, llk));

function logprob = lgmmprob(data, mu, sigma, w)
% compute the log probability of observations given the GMM
ndim = size(data, 1);
C = sum(mu.*mu./sigma) + sum(log(sigma));
D = (1./sigma)' * (data .* data) - 2 * (mu./sigma)' * data  + ndim * log(2 * pi);
logprob = -0.5 * (bsxfun(@plus, C',  D));
logprob = bsxfun(@plus, logprob, log(w));

function y = logsumexp(x, dim)
% compute log(sum(exp(x),dim)) while avoiding numerical underflow
xmax = max(x, [], dim);
y    = xmax + log(sum(exp(bsxfun(@minus, x, xmax)), dim));
ind  = find(~isfinite(xmax));
if ~isempty(ind)
    y(ind) = xmax(ind);
end

function [data, frate, feakind] = htkread(filename)
% read features with HTK format (uncompressed)
fid = fopen(filename, 'rb', 'ieee-be');
nframes = fread(fid, 1, 'int32'); % number of frames
frate   = fread(fid, 1, 'int32'); % frame rate in nano-seconds unit
nbytes  = fread(fid, 1, 'short'); % number of bytes per feature value
feakind = fread(fid, 1, 'short'); % 9 is USER
ndim = nbytes / 4; % feature dimension (4 bytes per value)
data = fread(fid, [ndim, nframes], 'float');
fclose(fid);
