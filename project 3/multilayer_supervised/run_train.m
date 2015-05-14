% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

%% Setup environment
% experiment information
% a struct containing network layer sizes etc
ei = [];

% add common directory to your path for
% minfunc and mnist data helpers
addpath ../common;
addpath(genpath('../common/minFunc_2012/minFunc'));

%% TODO: load face data
[train,test] = ex1_load_mnist(false);

% Add row of 1s to the dataset to act as an intercept term.
train.y = train.y+1; % make labels 1-based.
test.y = test.y+1; % make labels 1-based.

% Training set info
m=size(train.X,2);
n=size(train.X,1);

%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)


%TODO: decide proper hyperparameters.
% dimension of input features FOR YOU TO DECIDE
ei.input_dim = n;
% number of output classes FOR YOU TO DECIDE
ei.output_dim = 10;
% sizes of all hidden layers and the output layer FOR YOU TO DECIDE
ei.layer_sizes = [30 20 20 10 ei.output_dim];
% scaling parameter for l2 weight regularization penalty
ei.lambda = 1;
% which type of activation function to use in hidden layers
% feel free to implement support for different activation function
ei.activation_fun = 'logistic';
%ei.activation_fun = 'tanh';


%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% Gradient check
%x = gradient_check(@supervised_dnn_cost, params, 10, ei, train.X(:, 1:5000), train.y(1:5000), false)

%% setup minfunc options
options = [];
options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';
options.maxIter = 10;

%% run training
%{
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, train.X, train.y);
%}
% TODO:  1) check the gradient calculated by supervised_dnn_cost.m
%        2) Decide proper hyperparamters and train the network.
%        3) Implement SGD version of solution.
%        4) Plot speed of convergence for 1 and 3.
%        5) Compute training time and accuracy of train & test data.

%% Stochastic gradient descent
[opt_params, error] = stochastic_grad_desc(@supervised_dnn_cost, params, 0.01, 1, train.X, train.y, test.X, test.y, ei); 


%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, test.X, [], true);
[~,pred] = max(pred);
acc_test = mean(pred==test.y);
fprintf('test accuracy: %f\n', acc_test);


[~, ~, pred] = supervised_dnn_cost( opt_params, ei, train.X, [], true);
[~,pred] = max(pred);
acc_train = mean(pred==train.y);
fprintf('train accuracy: %f\n', acc_train);






