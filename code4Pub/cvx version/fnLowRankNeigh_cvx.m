function [R] = fnLowRankNeigh_cvx(X,lambda,k)
%%
% fnLowRankNeigh.m
% this function codes the manifold classification and reconstruction
% algorithm that attempts to ensure that each neighborhood has as low a
% rank as possible. The algorithm trades the rank of the neighborhood vs
% the nuclear norm of the weighted neighborhood matrix (this is a convex 
% relaxtion of the low-rank problem), the optimization problem being solved
% for each point is:-
%               ||x-Nw|| + lambda*||N1*diag(w)||*
% where x is the point in question, N its neighborhood, N1 the normalized
% neighborhood and w are the reconsrtuction weights
%
% This function uses the CVX toolbox for MATLAB available online at :
% http://cvxr.com/cvx to solve the convex optimization problem.
% ----------------------------------------------------------------------- %
% INPUT
% ----------------------------------------------------------------------- %
% X             : - the set of all points (nPts X nDims)
% embedDims     : - the number dimensions for the embedding
% nManifs       : - the number of manifolds/classes we are looking for
% lambda        : - the trade off parameter lambda
% k             : - number of neighbors
% ----------------------------------------------------------------------- %
% OUTPUT
% ----------------------------------------------------------------------- %
% mappedY       : - the embedded points
% class         : - the classification based on spectral clustering of the
%                   reconstruction matrix
% R             : - the set of partitioned reconstruction matrices.
%
% Author Name   : - Arun M Saranathan
% Date Created  : - 09/29/2015
% Date Modified : - 09/07/2016
% ----------------------------------------------------------------------- %

% find the size
[nDims, nPts] = size(X);

% intitalize reconstruction matrix and column of ones
R = zeros(nPts,nPts);
r1= ones(k,1);

% now find the neighborhoods
IDX = knnsearch(X',X','k',k+1);

% now for each point find the reconstruction coefficients that minimize the
% objective described above
for ii=1:nPts
    x               = X(:,ii);  % pick the point iteratively
    
    idx             = IDX(ii,2:end);
    N               = X(:,idx);      %select the neighborhood
    
    %now normalize the neighborhood
    N1              = N';
    meanN1          = mean(N1);
    meanN1          = repmat(meanN1,k,1) ;
    N1              = N1-meanN1;       %subtract mean
    N1              = N1';
    % now ensure each column is norm-1
    for i=1:k
        N1(:,i)     = N1(:,i)/norm(N1(:,i));
    end
    tic,
    cvx_begin quiet
        variable wt(k,1)
        M = N1*diag(wt);
    
        minimize ( (norm(x-N*wt,'fro')) + lambda*norm_nuc(M) )
        subject to 
            r1'*wt == 1;        
    cvx_end 
    clc
    toc
    % place these weights in the reconstruction matrix
    R(ii,idx)        = wt;
    fprintf('\n for point %d the error is %2.4f \n',ii,norm(x-N*wt,'fro'));
end %end-for

end %end-function
