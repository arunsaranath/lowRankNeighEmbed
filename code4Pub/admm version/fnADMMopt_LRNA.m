function [reconWeights] = fnADMMopt_LRNA(x,N,lambda, maxIter,beta,eps)
% fnADMMopt_LRNA
% this function implements the ADMM based update function for one point in
% the dataset and return the weights.
% ----------------------------------------------------------------------- %
% INPUT
% ----------------------------------------------------------------------- %
% x             : - the data-points (1 X nDims)
% N             : - the k-nearest neighbors (nDims X k)
% lambda        : - scaling of the rank w.r.t reconstruction error
% maxIter       : - maximum number of iterations before exit
% beta          : - the step-size (sort of)  
% eps           : - error tolerance
% ----------------------------------------------------------------------- %
% OUTPUT
% ----------------------------------------------------------------------- %
% reconWeights  : - the reocnstuction weight for each of the neighbors

%
% Author Name   : - Arun M Saranathan
% Date Created  : - 03/01/2016
% Date Modified : - 03/01/2016
% ----------------------------------------------------------------------- %

%normalize your neighborhood for this point
k                                  = size(N,2) ;
M                                  = N';
meanN1                             = mean(M);
meanN1                             = repmat(meanN1,k,1) ;
M                                  = M-meanN1;       %subtract mean
M                                  = M';
r1                                 = ones(k,1);
t1_old                             = Inf;
t2_old                             = Inf;

% now ensure each column is norm-1
for i=1:k
    M(:,i)                         = M(:,i)/norm(M(:,i));
end

E                                  = eye(k);
%find the constant P and Q matrices we defined earlier
Q = zeros(k,k); 
for i=1:k
    Q(i,i)                         = trace(E(:,i)*E(:,i)' * M'*M);
end


%intialize the variables we need
alpha                              = abs(rand(k,1));
alpha                              = alpha ./ sum(alpha);
Lambda_1                           = 0;
Lambda_2                           = zeros(length(x),length(alpha));
V                                  = zeros(size(Lambda_2));
P                                  = zeros(k,1);
% V_new                            = zeros(size(V));
% alpha_new                        = zeros(size(alpha));
% Lambda_1_new                     = zeros(size(Lambda_1));
% Lambda_2_new                     = zeros(size(Lambda_2));

for iter=1:maxIter
%     iter
    % write the update function of V
    temp                           = (M*diag(alpha)) - ((1/beta)*Lambda_2);
    [UU,S,VV]                      = svd(temp,'econ');
%     dbstop in fnSoftThresh_v1.m
    S_new                          = fnSoftThresh(S,(lambda/beta));
    V_new                          = UU*S_new*VV';
        
    %now update value of alpha
    F1                             = 1-((1/beta)*Lambda_1);
    F2                             = V_new+((1/beta)*Lambda_1);
    for i=1:k
        P(i,1)                     = trace(E(:,i)*E(:,i)' * M'*F2);
    end
    X                              = (2*N'*N) + (beta*r1'*r1*...
                                               eye(length(r1))) + (beta*Q);
    y                              = (2*N'*x) +(beta*r1*F1) + (beta*P);
    alpha_new                      = (inv(X'*X))*X'*y;
    
    %update the Lagrangian multipliers
    Lambda_1_new                   =  Lambda_1 + beta*(r1'*alpha_new- 1);
    Lambda_2_new                   =  Lambda_2 + beta*(V_new - (M*diag(alpha_new)));
    
    % if the Lagrangian multipliers are 0 at convergence
    t1                             = norm(Lambda_1_new-Lambda_1,'fro')...
                                                     /norm(Lambda_1,'fro');
    t2                             = norm(Lambda_2_new-Lambda_2,'fro')...
                                                     /norm(Lambda_2,'fro');
                                                 
    t3(iter)            = norm(alpha_new-alpha)/norm(alpha);    
    t4(iter)            = norm(V_new-V)/norm(V);

    temp1(iter) = t1;
    temp2(iter) = t2;
    
    reconErr(iter)  = norm(x-N*alpha_new);
    nucNormErr(iter) = sum(sum(S_new));
    
    totalErr(iter) = reconErr(iter) + (lambda*nucNormErr(iter));
     
    if((t2<=eps)&&(t1<=eps))
        break
    end
    
    if((t1>t1_old)||(t2>t2_old))
%         break;
    else
        %change to the newer version
        Lambda_1                       = Lambda_1_new;
        Lambda_2                       = Lambda_2_new;
        V                              = V_new;
        alpha                          = alpha_new;
        t1_old                         = temp1(iter);
        t2_old                         = temp1(iter);
    end
end %end-for

reconWeights = alpha;
end %end-function
