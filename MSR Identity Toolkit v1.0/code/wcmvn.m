function Fea = wcmvn(fea, win, varnorm)
% performs cepstral mean and variance normalization over a sliding window
%
% Inputs:
%   - fea     : input ndim x nobs feature matrix, where nobs is the 
%				number of frames and ndim is the feature dimension
%   - win     : length of the sliding window (should be an odd number)
%   - varnorm : binary switch (false|true), if true variance is normalized 
%               as well
% Outputs:
%   - Fea     : output ndim x nobs normalized feature matrix.
%
% Omid Sadjadi <s.omid.sadjadi@gmail.com>
% Microsoft Research, Conversational Systems Research Center

if ( nargin < 3 ), 
    varnorm = false;
end
if ( nargin == 1),
    win = 301;
end

if ( mod(win, 2) == 0 ),
    fprintf(1, 'Error: Window length should be an odd number!\n');
    return;
end

[ndim, nobs]  = size(fea);
if ( nobs < win ),
    if ~mod(nobs, 2)
        nobs = nobs+1;
        fea = [fea, fea(:, end)];
    end
    win = nobs;
end

epss = 1e-20;
Fea = zeros(ndim, nobs);
idx = 1 : ( win - 1 )/2;
Fea(:, idx) = bsxfun(@minus, fea(:, idx), mean(fea(:, 1 : win), 2));

if varnorm,
    Fea(:, idx) = bsxfun(@rdivide, Fea(:, idx), std(fea(:, 1 : win), [], 2) + epss);

    for m = ( win - 1 )/2 + 1 : nobs - ( win - 1 )/2
        idx = m - ( win - 1 )/2 : m + ( win - 1 )/2;
        Fea(:, m) = ( fea(:, m) - mean(fea(:, idx), 2) )./(std(fea(:, idx), [], 2) + epss);
    end
else
    for m = ( win - 1 )/2 + 1 : nobs - ( win - 1 )/2
        idx = m - ( win - 1 )/2 : m + ( win - 1 )/2;
        Fea(:, m) = ( fea(:, m) - mean(fea(:, idx), 2) );
    end
end

idx = (nobs - ( win - 1 )/2 + 1) : nobs;
Fea(:, idx) = bsxfun(@minus, fea(:, idx), mean(fea(:, nobs-win+1:nobs), 2));

if varnorm,
    Fea(:, idx) = bsxfun(@rdivide, Fea(:, idx), std(fea(:, nobs-win+1:nobs), [], 2) + epss);
end
