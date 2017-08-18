function w = hamming(L)
% hamming window of length L
%
% Omid Sadjadi <s.omid.sadjadi@gmail.com>
% Microsoft Research, Conversational Systems Research Center

w = 0.54- 0.46 * cos(2 * pi * ( 0 : L - 1 )/( L - 1 ) );
