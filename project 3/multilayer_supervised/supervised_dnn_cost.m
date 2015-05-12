function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation

%% Determine activation function type.
if strcmp(ei.activation_fun, 'logistic')
    f            = @sigmoid_activation;
    f_derivative = @(A) (A.*(1-A));
elseif strcmp(ei.activation_fun, 'tanh')
    f            = @tanh_activation;
    f_derivative = @(A) ( 1-A.^2); 
end

%% Characteristics of data.
[d, m] = size(data);
% Determine the number of unique classes
K = length(unique(labels));

%% Default values.
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% Reshape into network.
stack     = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct      = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);

%% Forward propagation.


%{ 
act=obj.network_input.features;
 afunc=obj.network_design.activationFunction;
 
 obj.network_output.stack
 for i=1:length(obj.network_output.stack)
     this_weight = obj.network_output.stack{i};
     act = afunc(this_weight.W*act + this_weight.b);
     obj.network_output.activations{i}=act;
 end
%}                
%TODO:  Need to fill out hAct, activations at each layer
%       i.e. hAct{l+1}.activation = Z
%TODO:  Need to return probability from forward propagation
%       Z_output = ...
%       i.e. probability = softmax(Z_output)

%% Return here if only predictions (po == True) desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% Compute cost.
cost = network.lossfunc;

%% Compute gradients using backpropagation algorithm.
deltas       = cell(numHidden + 1, 1);

% Compute the delta matrix for output layer
I = eye(K);
output_index = numHidden + 1;
deltas{output_index}.delta_matrix = probability - I(:, labels); 

% Compute the delta matrices for hidden layers, h hidden layers
for h = numHidden: -1 : 1
    deltas{h}.delta_matrix =  (stack{h + 1}.weight_matrix' * deltas{h + 1}.delta_matrix) .* f_derivative(hAct{h + 1}.activation);
end

% Compute the gradients
for h = 1:(numHidden + 1)
    % Gradients for weight_matrix
    gradStack{h}.weight_matrix = deltas{h}.delta_matrix * hAct{h}.activation';
    
    % Gradients for bias
    gradStack{h}.bias_vector = sum(deltas{h}.delta_matrix, 2);
end


%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



