function Fea = cmvn(fea, varnorm)
% performs cepstral mean and variance normalization
%
% Inputs:
%   - fea     : input ndim x nobs feature matrix, where nobs is the 
%				number of frames and ndim is the feature dimension
%   - varnorm : binary switch (false|true), if true variance is normalized 
%               as well
% Outputs:
%   - Fea     : output p x n normalized feature matrix.
%
% Omid Sadjadi <s.omid.sadjadi@gmail.com>
% Microsoft Research, Conversational Systems Research Center

if ( nargin == 1 ), varnorm = false; end 
    
mu = mean(fea, 2);
if varnorm,
    stdev = std(fea, [], 2);
else
    stdev = 1;
end

Fea = bsxfun(@minus, fea, mu);
Fea = bsxfun(@rdivide, Fea, stdev);
