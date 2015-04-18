% Gradient descent with fixed learning parameter

function [theta, f ] = grad_desc_naive( funObj, theta0, options, X, y )
%GRAD_DESC Summary of this function goes here
%   Detailed explanation goes here
theta = theta0;    

eta   = 1;
N     = 150000;

for i = 1:N
    
    [f  , g] = funObj(theta, X, y);
    
    if mod(i,10000) == 0
        eta = eta/2;
    end    
    
    theta = theta - eta * g/norm(g);    
end 
    
end
