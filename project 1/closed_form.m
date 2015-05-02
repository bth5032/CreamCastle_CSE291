function [theta, f, g, exitflag ] = closed_form( X, y )
%CLOSED_FORM The closed form solution to the linear inverse problem y = X*theta + e.
%If we assume there are more data-points than parameters, it is safe to
%assume X has full row rank -> X*X' is positive definite and invertible.
%Thus the solutions to the normal equations: 2Xy' = 2X*X'*theta is
%immediate. 

theta=(X*X')\((X)*(y'));

end
