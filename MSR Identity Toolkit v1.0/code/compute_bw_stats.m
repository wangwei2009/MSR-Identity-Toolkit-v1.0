function [N, F] = compute_bw_stats(feaFilename, ubmFilename, statFilename)
% extracts sufficient statistics for features in feaFilename and GMM 
% ubmFilename, and optionally save the stats in statsFilename. The 
% first order statistics are centered.
%
% Inputs:
%   - feaFilename  : input feature file name (string) or a feature matrix 
%					(one observation per column)
%   - ubmFilename  : file name of the UBM or a structure with UBM 
%					 hyperparameters.
%   - statFilename : output file name (optional)   
%
% Outputs:
%   - N			   : mixture occupation counts (responsibilities) 
%   - F            : centered first order stats
%
% References:
%   [1] N. Dehak, P. Kenny, R. Dehak, P. Dumouchel, and P. Ouellet, "Front-end 
%       factor analysis for speaker verification," IEEE TASLP, vol. 19, pp. 788-798,
%       May 2011. 
%   [2] P. Kenny, "A small footprint i-vector extractor," in Proc. Odyssey, 
%       The Speaker and Language Recognition Workshop, Jun. 2012.
%
%
% Omid Sadjadi <s.omid.sadjadi@gmail.com>
% Microsoft Research, Conversational Systems Research Center

if ischar(ubmFilename),
	tmp  = load(ubmFilename);
	ubm  = tmp.gmm;
elseif isstruct(ubmFilename),
	ubm = ubmFilename;
else
    error('Oops! ubmFilename should be either a string or a structure!');
end
[ndim, nmix] = size(ubm.mu);
m = reshape(ubm.mu, ndim * nmix, 1);
idx_sv = reshape(repmat(1 : nmix, ndim, 1), ndim * nmix, 1);

if ischar(feaFilename),
    data = htkread(feaFilename);
else
    data = feaFilename;
end

[N, F] = expectation(data, ubm);
F = reshape(F, ndim * nmix, 1);
F = F - N(idx_sv) .* m; % centered first order stats

if ( nargin == 3)
	% create the path if it does not exist and save the file
	path = fileparts(statFilename);
	if ( exist(path, 'dir')~=7 && ~isempty(path) ), mkdir(path); end
	parsave(statFilename, N, F);
end

function parsave(fname, N, F) %#ok
save(fname, 'N', 'F')

function [N, F] = expectation(data, gmm)
% compute the sufficient statistics
post = postprob(data, gmm.mu, gmm.sigma, gmm.w(:));
N = sum(post, 2);
F = data * post';

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
