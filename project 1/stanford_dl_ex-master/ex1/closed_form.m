function [theta, f, g, exitflag ] = closed_form( X, y )
%CLOSED_FORM Summary of this function goes here
%   Detailed explanation goes here
theta=(X*X')\((X)*(y'));
[f, g]=linear_regression(theta, X, y); 
exitflag=1;
end

