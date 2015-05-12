function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation

%% Determine activation function type.
if strcmp(ei.activation_fun, 'logistic')
    func = @sigmoid_activation;
    grad = @(A) (A.*(1-A));
elseif strcmp(ei.activation_fun, 'tanh')
    func = @tanh_activation;
    grad = @(A) ( 1-A.^2); 
end

%% Default values.
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% Reshape into network.
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);

%% Forward prop
network.forwardProp; 

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% Compute loss
loss = 0
% Sum over number of examples
for i=1:

cost = network.lossfunc;

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



