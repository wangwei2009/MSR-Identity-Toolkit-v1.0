function [V, D] = lda(data, labels)
% performs linear discriminant analysis (LDA) using the Fisher criterion,
% that is maximize the between class variation while reducing the within
% class variance. Each column in the data matrix is an observation.
%
% Technically, the Fisher criterion to be maximized is in the form:
%                           
%                         V'.Sb.V           
%                 J(V) = ----------
%                         V'.Sw.V
%
% which is a Rayleigh quotient, therefore the solution, V, is the 
% generalized eigenvectors of Sb.V = D.Sw.V.
%
% Inputs:
%   - data        : input data matrix, one observation per column
%   - labels      : class labels for observations in data matrix  
%
% Outputs:
%   - V           : LDA transformation matrix that maximizes the
%                   discrimination among different classes
%   - D           : Eigenvalues of the generalized EVD for LDA 
%
% References:
%   [1] K. Fukunaga, Introduction to Statistical Pattern Recognition, 2nd ed.
%       New York: Academic Press, 1990, ch. 10. 
%
%
% Omid Sadjadi <s.omid.sadjadi@gmail.com>
% Microsoft Research, Conversational Systems Research Center

[ndim, nobs] = size(data);

if ( nobs ~= length(labels) ),
	error('oh dear! number of data samples should match the number of labels!');
end

data = bsxfun(@minus, data, mean(data, 2)); % centering the data
Sw = data * data'; % assume the same covariance for all classes

classes = unique(labels);
nclasses = length(classes);
mu_c = zeros(ndim, nclasses);
for class = 1 :  nclasses,
    idx = find(ismember(labels, classes(class)));
    mu_c(:, class) = sqrt(length(idx)) * mean(data(:, idx), 2);
end

Sb = mu_c * mu_c';

[V, D] = eig(Sb, Sw); % generalized EVD 
[D, I] = sort(diag(D), 1, 'descend');
V = V(:, I);

% the effective dimensionality of the transformation matrix
Vdim = min(size(V, 2), nclasses - 1);
V = V(:, 1 : Vdim);
D = D(1 : Vdim);

% normalize the eigenvalues and eigenvectors
D = D/sum(D);
V = length_norm(V);
