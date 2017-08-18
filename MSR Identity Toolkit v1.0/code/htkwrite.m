function htkwrite(filename, data, frate, feakind)
% write features with HTK format
%
% Omid Sadjadi <s.omid.sadjadi@gmail.com>
% Microsoft Research, Conversational Systems Research Center

[ndim, nframes] = size(data);
fid = fopen(filename, 'wb', 'ieee-be');
fwrite(fid, nframes, 'int32'); % number of frames
fwrite(fid, frate, 'int32'); % frame rate in 100 nano-seconds unit
fwrite(fid, 4 * ndim, 'short'); % 4 bytes per feature value
fwrite(fid, feakind, 'short'); % 9 is USER
fwrite(fid, data, 'float');
fclose(fid);
