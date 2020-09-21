function embedPts = dataEmbed_v2(R,nDims)
%%
%dataEmbed.m
% This function embeds the data points into different dimension using the
% same procedure as LLE or LRE. 
% ------------------------------------------------------------------------%
% Inputs
% ------------------------------------------------------------------------%
% R                         : - the reconstruction matrix
% nDims                     : - dimensionality of space we embed in
% ---------------------------------------------------------------------------- %
% Ouputs
% ---------------------------------------------------------------------------- %
% embedPts                  : - the embedded points
% ---------------------------------------------------------------------------- %
%
% This is a initial model and other initializations are required in the final
% case.
%
%
% Arun M Saranathan
% 05/29/2014
% ---------------------------------------------------------------------------- %

%find size of R
[~,n] = size(R);

%now use LLE embedding as
M = eye(n) - R;
M = M' * M;
eps = 1e-6;
%now perfrom eigenvalue decomposition
[V, D] = eig(M + eps * eye(n));
% eigVal = diag(D);
% [srtEigVal,indx] = sort(eigVal,'ascend') ;

%now return the largest ones
embedPts = V(:,2:nDims+1);

end