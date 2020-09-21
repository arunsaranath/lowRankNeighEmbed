clear all
close all
clc
%% testLRNE.m
% This code file is used to test the LRNE algorithm on a 3D toy dataset
% with 2 manifolds. There are two variant of the LRNE algorithm available
% in this toolbox:
%   1) CVX version: This version uses the CVX toolbox to solve the LRNE
%   optimization and requires the the CVX toolbox for MATLAB available 
%   online @ :http://cvxr.com/cvx.
%   2) ADMM solution: In this case the optimization problem is solved using
%   the ADMM based solution described in the paper.

load test_dataSet.mat
%% DISPLAY THE MANIFOLDS IN THE TEST DATASET- TRUE LABELS
figure('Color', [1,1,1])
scatter3(totalPts(:,1),totalPts(:,2),totalPts(:,3),100,labels,'.')
axis equal
title('Original Manifold membership')
%% PERFORM LRNE TO GET THE LRNE RECONSTRUCTION MATRIX
lambda = 1e-1;
k = 30;
% The CVX version
% [R] = fnLowRankNeigh_cvx(totalPts',lambda,k, 1e-4, 1e-6, 1e2);
% The ADMM version
[R] = fnLowRankNeigh_ADMM(totalPts',lambda,k, 1e-4, 1e-6, 1e2);    
%% PERFORM THE CLUSTERING TO GET THE MISCLASSIFICATION RATE

% extract the adjacency matrix 
%(note this is not the exact adjacency matrix defined in the paper but a 
% quick version for illustration purposes for best results use the version
% described in the paper)
W = max(abs(R'),abs(R));

% calculate the symmetrized Laplacian matrix
D = diag(sum(W));
L = D - W;
L_sym = eye(size(D)) - (inv(D.^0.5)*W*inv(D.^0.5));
% calculate the eigne vectors
[eigVec,eigVal] = eig(L_sym);

figure('Color' ,[1,1,1])
subplot(1,2,1)
plot(eigVec(:,2))
subplot(1,2,2)
plot(eigVec(:,3))
%%
% threshold to perform clustering
estLabels = zeros(size(eigVec(:,2)));
estLabels(eigVec(:,2)<0) = 1;
estLabels(eigVec(:,2)>=0) = 2;
% the misclassification rate in percentage
classPerf = sum(abs(labels - estLabels))/length(labels)*100;
classPerf = min(classPerf, (100-classPerf));
fprintf('The misclassification rate (in percent) is %1.3f\n', classPerf);

%% DISPLAY THE MANIFOLDS IN THE TEST DATASET- estimated labels
figure('Color', [1,1,1])
scatter3(totalPts(:,1),totalPts(:,2),totalPts(:,3),100,estLabels,'.')
axis equal
title('Estimated Manifold membership')
%% PERFORM EMBEDDING
[Y] = dataEmbed_v2(R,2);

% GET THE INDICIES OF EACH CLASS
i1 = find(estLabels ==1);
i2 = find(estLabels ==2);

% DISPLAY THE EMBEDDING
figure(6)
set(gcf,'Color',[1 1 1])
subplot(1,2,1)
scatter(Y(i1,1),Y(i1,2),150,'b.')
set(gca,'XTickLabel','','YTickLabel','')
% view([33 90])
% axis equal
subplot(1,2,2)
scatter(Y(i2,1),Y(i2,2),150,'r.')
set(gca,'XTickLabel','','YTickLabel','')