function Fea = fea_warping(fea, win)
% performs feature warping on feature streams over a sliding window
%
% Inputs:
%   - fea     : input ndim x nobs feature matrix, where nobs is the 
%				number of frames and ndim is the feature dimension
%   - win     : length of the sliding window (should be an odd number)
%
% Outputs:
%   - Fea     : output ndim x nobs normalized feature matrix.
%
% Omid Sadjadi <s.omid.sadjadi@gmail.com>
% Microsoft Research, Conversational Systems Research Center

if ( nargin == 1),
    win = 301;
end

if ( mod(win, 2) == 0 ),
    fprintf(1, 'Error: Window length should be an odd number!\n');
    return;
end

fea = fea';
[nobs, ndim]  = size(fea);

if ( nobs < win ),
    if ~mod(nobs, 2)
        nobs = nobs + 1;
        fea = [fea; fea(end, :)];
    end
    win = nobs;
end

Fea = zeros(nobs, ndim);
[~, R] = sort(fea(1 : win, :));
[~, R] = sort(R);
arg = ( R(1 : ( win - 1 ) / 2, :) - 0.5 ) / win;
Fea(1 : ( win - 1 ) / 2, :) = norminv(arg, 0, 1);
for m = ( win - 1 ) / 2 + 1 : nobs - ( win - 1 ) / 2
    idx = m - ( win - 1 ) / 2 : m + ( win - 1 ) / 2;
    foo = fea(idx, :);
    R = sum(bsxfun(@lt, foo, foo(( win - 1 ) / 2 + 1, :))) + 1; % get the ranks
    arg = ( R - 0.5 ) / win;    % get the arguments
    Fea(m, :) = norminv(arg, 0, 1); % transform to normality
end
[~, R] = sort(fea(nobs - win + 1 : nobs, :));
[~, R] = sort(R);
arg = ( R( ( win + 1 ) / 2 + 1 : win, :) - 0.5 ) / win;
Fea(nobs - ( win - 1 ) / 2 + 1 : nobs, :) = norminv(arg, 0, 1);
Fea = Fea';
