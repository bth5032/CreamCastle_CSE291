function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
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
% Initial activation is simply the data input
hAct{1}.activation = data;

% Using prior layers, compute activations
for h = 1:numHidden
    Z = stack{h}.weight_matrix * hAct{h}.activation + stack{h}.bias_vector;
    Z = f(Z);
    hAct{h+1}.activation = Z;
end

% Output layer
H = numHidden + 1;
Z_output = stack{H}.weight_matrix * hAct{H} + stack{H}.bias_vector;
probability = softmax(Z_output);


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


%% Compute weight penalty cost and gradient for non-bias terms.
%%% YOUR CODE HERE %%%


%% Reshape gradients into vector.
[grad] = stack2params(gradStack);
end



