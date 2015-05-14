%MATT: since minfunc only takes a scalar cost, we take cost at the output and pass delta as the gradient 
function [ cost, grad, probability, hAct, delta] = supervised_dnn_cost( theta, ei, data, labels, pred_only )
%% supervised_dnn_cost Does all the work of cost / gradient computation
%pred_only==1: get hAct
%pred_only==2: get hAct, delta
delta=-1;

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
end

% Reshape into network.
stack     = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct      = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);


%% Forward propagation
% Initial activation is the data input
hAct{1}.activation = data;

% Using prior layers, compute activations
for h = numHidden: -1 :1
    Z = stack{h}.W * hAct{h}.activation + stack{h}.b;
    Z = f(Z);
    hAct{h+1}.activation = Z(:);
end

% Output layer
H = numHidden + 1;
Z_output = stack{H}.W * hAct{H}.activation + stack{H}.b;
probability = softmax(Z_output);
cost=norm(probability); 

% Return hAct if pred_only==1
if po==1
    return;
end

%% Compute delta based on current activations

deltaStack = cell(numHidden + 1, 1);
% Compute the delta matrix for output layer

output_index = numHidden + 1;
I=eye(length(unique(labels)));
err = (probability - diag(I(unique(labels), : )));

z = stack{output_index}.W * hAct{output_index}.activation(:) + stack{output_index}.b(:);
deltaStack{output_index}.W = -err.*f_derivative(z);
deltaStack{output_index}.b = zeros(length(unique(labels)),1); 

% Compute the delta matrices for hidden layers, h hidden layers
for h = (output_index-1): -1: 1
    this_z = stack{h}.W * hAct{h}.activation(:) + stack{h}.b(:);
    weight_loss=(stack{h+1}.W' * deltaStack{h+1}.W );
    bias_loss=(stack{h+1}.b' * deltaStack{h+1}.b) ;
    
    deltaStack{h}.W =  weight_loss .* f_derivative(this_z); 
    deltaStack{h}.b = bias_loss .* f_derivative(this_z);
end

cost= -( ( ones(K,1) - probability)'*log(ones(K,1)- f(z)') + probability'*log(f(z)'));

%Return delta, hAct if pred_only==2
if po==2
    delta = stack2params(deltaStack);
    return;
end


%% Collect outputs
for h=1:numHidden + 1
    % Gradients for weight_matrix
    gradStack{h}.W = (hAct{h}.activation * deltaStack{h}.W')';
    
    % Gradients for bias
    gradStack{h}.b = deltaStack{h}.b;
end

grad = stack2params(gradStack);

end



