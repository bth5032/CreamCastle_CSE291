function [f, g] = linear_regression(theta, X, y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %   X - The feature data stored in a matrix. X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example.  y(j) is the target for example j.
  %

  f=norm(y'-X'*theta); % Euclidean norm between targets and guess
  g=X*X'*theta-(y*X')'; % Gradient of the objective function f with respect to our weight vector
