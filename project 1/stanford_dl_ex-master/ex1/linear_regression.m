function [f, g] = linear_regression(theta, X, y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %   X - The feature data stored in a matrix. X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example.  y(j) is the target for example j.
  %

  e=y'-X'*theta;
  f=e'*e; % Euclidean norm squared between targets and guess
  g=2*(X*X')*theta-2*(y*X')'; % Gradient of the objective function f with respect to our weight vector
