%Implement Gauss-Newton algorithm for gradient descent.

function [theta, f, g, exitflag ] = grad_desc( funObj, theta0, options, X, y )
%GRAD_DESC Summary of this function goes here
%   Detailed explanation goes here
theta=theta0;
J=-X';

g=Inf;
tol=1e-7;
n=1;

while 1
    [f, g]=funObj(theta, X, y);
    if g<tol
        disp(['grad_desc: converged after ' num2str(n) ' iterations']);
        exitflag=1;
        break;
    end
    
    if ~isempty(options)
        s = fieldnames(options);
        
        if ~isempty(intersect(s, 'MaxIter'))
            if n > options.MaxIter;
                disp(['grad_desc: did not converge after' n 'iterations']);
                exitflag=0;
            end
        end
        
        
    else
        options.MaxIter=100;
    end
    
    
    theta=theta-(J'*J)\J'*(y'-X'*theta);
end
end
