function [x_thresh] = fnSoftThresh(x,eps)
%fnSoftThresh.m
% this function applies the soft thersholding operator to a vector. This
% function can be defined as:-
%                   | x - e    if x > e
%          S_e(c) = | x + e    if x < -e
%                   | 0        otherwise
% ----------------------------------------------------------------------- %
% INPUT
% ----------------------------------------------------------------------- %
% x             : - the vector to be thresholded
% eps           : - control parameter, i.e. region where the threshold
%                   applies
% ----------------------------------------------------------------------- %
% OUTPUT
% ----------------------------------------------------------------------- %
% x_thresh      : - the soft thresholded vector
%
%
% Author Name   : - Arun M Saranathan
% Date Created  : - 03/01/2016
% Date Modified : - 03/01/2016
% ----------------------------------------------------------------------- %

% find the sign of the vector
x                                 = diag(x);
% now threshold the vector
temp                              = (x - eps);
temp(temp<0  )                    = 0;
x_thresh                          = temp;
x_thresh                          = diag(x_thresh');
end %end-function