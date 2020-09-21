function [R] = fnLowRankNeigh_ADMM(X,lambda,k, beta, epsilon,maxIter)
% fnLowRankNeigh_ADMM.m
% this function codes the manifold classification and reconstruction
% algorithm that attempts to ensure that each neighborhood has as low a
% rank as possible. The algorithm trades the rank of the neighborhood vs
% the nuclear norm of the weighted neighborhood matrix (this is a convex 
% relaxtion of the low-rank problem), the optimization problem being solved
% for each point is:-
%               argmin ||x-Nw|| + lambda*||M*diag(w)||*
%                 w
%           subject to   r1'*wt = 1
% where x is the point in question, N its neighborhood, M the normalized
% neighborhood and w are the reconsrtuction weights. The optimization is
% done using an Alternating Direction of Multipliers Methods (ADMM)
%
% ----------------------------------------------------------------------- %
% INPUT
% ----------------------------------------------------------------------- %
% X             : - the set of all points (nPts X nDims)
% embedDims     : - the number dimensions for the embedding
% nManifs       : - the number of manifolds/classes we are looking for
% lambda        : - the trade off parameter lambda
% k             : - number of neighbors
% beta          : - the step size for each optimization
% epsilon       : - the error tolerance
% maxIter       : - maximum number of iterations for each points
%                   minimization
% ----------------------------------------------------------------------- %
% OUTPUT
% ----------------------------------------------------------------------- %
% mappedY       : - the embedded points
% class         : - the classification based on spectral clustering of the
%                   reconstruction matrix
% R             : - the set of partitioned reconstruction matrices.
%
% Author Name   : - Arun M Saranathan
% Date Created  : - 03/01/2016
% Date Modified : - 03/01/2016
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
    x               = X(:,ii);       % pick the point iteratively    
    idx             = IDX(ii,2:end); % select the neighborhood   
    N               = X(:,idx);  
    
     tic,
     [reconWeights] = fnADMMopt_LRNA(x,N,lambda, maxIter,beta,epsilon);
     toc
    
    R(ii,idx)      = reconWeights;
end %end-for

end %end-function