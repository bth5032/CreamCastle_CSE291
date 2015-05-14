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


%% Load MNIST Data
% [train,test] = ex1_load_mnist(false);
% 
% % Add row of 1s to the dataset to act as an intercept term.
% train.y = train.y+1; % make labels 1-based.
% test.y = test.y+1; % make labels 1-based.
% 
% train_X = train.X;
% train_y = train.y;
% test_X  = test.X;
% test_y  = test.y;
% 
% % Training set info
% m = size(train.X,2);
% n = size(train.X,1);
% K = length(unique(test.y));

%% Load the NimStim Data
train_X = Final_NimStim_Input_Matrix;
train_y = Final_NimStim_Targets;

test_X = Final_NimStim_Input_Matrix;
test_y = Final_NimStim_Targets;

%Training set info
m = size(train_X,2);
n = size(train_X,1);

K = length(unique(train_y));



% data = Final_NimStim_Input_Matrix;
% labels = Final_NimStim_Targets;
% 
% i = randperm(length(labels));
% data   = data(:,i);
% labels = labels(i); 
% 
% % Partition data
% train_X = data(:,1:255);
% train_y = labels(1:255);
% 
% test_X = data(:,256:341);
% test_y = labels(256:341);
% 
% % Training set info
% m = size(train_X,2);
% n = size(train_X,1);
% 
% K = length(unique(train_y));


%% Load the POFA Data
% train_X = Final_POFA_Input_Matrix;
% train_y = Final_POFA_Targets;
% 
% test_X = Final_POFA_Input_Matrix;
% test_y = Final_POFA_Targets;
% 
% % Training set info
% m = size(train_X,2);
% n = size(train_X,1);
% 
% K = length(unique(train_y));


%% Populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)


% Decide proper hyperparameters.
% dimension of input features FOR YOU TO DECIDE
ei.input_dim = n;
% number of output classes FOR YOU TO DECIDE
ei.output_dim = K;
% sizes of all hidden layers and the output layer FOR YOU TO DECIDE
ei.layer_sizes = [30, 20, ei.output_dim];
% scaling parameter for l2 weight regularization penalty
ei.lambda = 1;
% which type of activation function to use in hidden layers
% feel free to implement support for different activation function
ei.activation_fun = 'logistic';
%ei.activation_fun = 'tanh';


%% Setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);


%% Setup minfunc options
options = [];
options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';
options.maxIter = 1000;

%% Training
t_min = cputime;
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, train_X, train_y);
e_min = cputime - t_min

% TODO:  1) check the gradient calculated by supervised_dnn_cost.m
%        2) Decide proper hyperparamters and train the network.
%        3) Implement SGD version of solution.
%        4) Plot speed of convergence for 1 and 3.
%        5) Compute training time and accuracy of train & test data.

%% Training with stochastic gradient descent
% t_sgd = cputime;
% [opt_params, error] = stochastic_grad_desc(@supervised_dnn_cost, params, 0.01, 100, train_X, train_y, test_X, test_y, ei); 
% e_sgd = cputime - t_sgd

%% Accuracy on test and train set.
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, test_X, [], true);
[~,pred] = max(pred);
acc_test = mean(pred==test_y);
fprintf('test accuracy: %f\n', acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, train_X, [], true);
[~,pred] = max(pred);
acc_train = mean(pred==train_y);
fprintf('train accuracy: %f\n', acc_train);





