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

%{
% Backtracking algorithm parameters
alpha = 0.15;
beta  = 0.5;
eta   = 1;
N     = 100000;

for i = 1:N
    
    [f  , g] = funObj(theta, X, y);
    [f_n,g]  = funObj(theta-eta*g,X,y);
    
    while f_n >= f - alpha*eta*norm(g) 
        eta = eta*beta;
        if eta < 1e-10
            break
    end    
    
    theta = theta - eta * g/norm(g);
    disp(sqrt(f/800));
    
end
%}    
    
end
