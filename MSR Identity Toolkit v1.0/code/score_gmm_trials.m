function llr = score_gmm_trials(models, testFiles, trials, ubmFilename)
% computes the log-likelihood ratio of observations given the UBM and
% speaker-specific (MAP adapted) models. 
%
% Inputs:
%   - models      : a cell array containing the speaker specific GMMs 
%   - testFiles   : a cell array containing feature matrices or file names
%   - trials      : a two-dimensional array with model-test verification
%                   indices (e.g., (1,10) means model 1 against test 10)
%   - ubmFilename : file name of the UBM or a structure containing 
%					the UBM hyperparameters that is,
%					(ubm.mu: means, ubm.sigma: covariances, ubm.w: weights)
%
% Outputs:
%   - llr		  : log-likelihood ratios for all trials (one score per trial)
%
%
% Omid Sadjadi <s.omid.sadjadi@gmail.com>
% Microsoft Research, Conversational Systems Research Center

if ~iscell(models),
    error('Oops! models should be a cell array of structures!');
end

if ischar(ubmFilename),
	tmp = load(ubmFilename);
	ubm = tmp.gmm;
elseif isstruct(ubmFilename),
	ubm = ubmFilename;
else
	error('oh dear! ubmFilename should be either a string or a structure!');
end

if iscellstr(testFiles),
    tlen = length(testFiles);
    tests = cell(tlen, 1);
    for ix = 1 : tlen,
        tests{ix} = htkread(testFiles{ix});
    end
elseif iscell(testFiles),
    tests = testFiles;
else
    error('Oops! testFiles should be a cell array!');
end

ntrials = size(trials, 1);

llr = zeros(ntrials, 1);
parfor tr = 1 : ntrials,
    gmm = models{trials(tr, 1)};
    fea = tests{trials(tr, 2)};
    ubm_llk = compute_llk(fea, ubm.mu, ubm.sigma, ubm.w(:));
    gmm_llk = compute_llk(fea, gmm.mu, gmm.sigma, gmm.w(:));
    llr(tr) = mean(gmm_llk - ubm_llk);
end

function llk = compute_llk(data, mu, sigma, w)
% compute the posterior probability of mixtures for each frame
post = lgmmprob(data, mu, sigma, w);
llk  = logsumexp(post, 1);

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
