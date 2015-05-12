function [ cost, grad, pred_prob, delta] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%% supervised_dnn_cost Does all the work of cost / gradient computation

%Preprocessing
% Determine activation function type.
if strcmp(ei.activation_fun, 'logistic')
    f            = @NetworkDesign.logisticFunc;
    f_derivative = @(A) (A.*(1-A));
elseif strcmp(ei.activation_fun, 'tanh')
    f            = @tanh;
    f_derivative = @(A) ( 1-A.^2); 
end

% Characteristics of data.
[d, m] = size(data);
% Determine the number of unique classes
K = length(unique(labels));

% Default values.
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

% Reshape into network.
stack     = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct      = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);


%% Forward propagation
% Initial activation is simply the data input
hAct{1}.activation = data;

% Using prior layers, compute activations
for h = 1:numHidden
    Z = stack{h}.W * hAct{h}.activation + stack{h}.b;
    Z = f(Z);
    hAct{h+1}.activation = Z;
end

% Output layer
H = numHidden + 1;
Z_output = stack{H}.W * hAct{H} + stack{H}.b;
probability = softmax(Z_output);

% Return here if only predictions (po == True) desired
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% Compute cost, delta and gradient based on current activations
deltaStack       = cell(numHidden + 1, 1);
costStack       = cell(numHidden + 1, 1);

% Compute the delta matrix for output layer
I = eye(K);
output_index = numHidden + 1;
deltaStack{output_index}.W = probability - I(:, labels); 
deltaStack{output_index}.b = 0;

% Compute the delta matrices for hidden layers, h hidden layers
for h = numHidden : -1 : 1 
    deltaStack{h}.W =  (stack{h + 1}.W' * deltaStack{h + 1}.W) .* f_derivative(hAct{h + 1}.activation);
    deltaStack{h}.b =  (stack{h + 1}.b' * deltaStack{h + 1}.b) .* f_derivative(hAct{h + 1}.activation);
end

%Compute the gradient, cost, starting from ouput layer
%MATT: minFunc or GradientDescent will take care of the actual weight updates.
%Our job in this function is just cost and gradient for the current weights
%and activations. 
for h = 1 : -1 : (numHidden + 1)
    
    %Gradient for weight_matrix
    gradStack{h}.W = f_derivative(hAct{h}.activation);
    
    %Gradient for bias
    gradStack{h}.b = f_derivative(stack{h}.b);
    
    %Compute weight cost 
    costStack{h}.W = deltaStack{h+1}.W'*stack{h}.W; 
    
    %Compute bias cost 
    costStack{h}.W = deltaStack{h+1}.b'*stack{h}.b; 
end


%% Collect outputs
grad = stack2params(gradStack);
delta = stack2params(deltaStack); 
cost = stack2params(costStack); 
pred_prob = hAct{end};
end



