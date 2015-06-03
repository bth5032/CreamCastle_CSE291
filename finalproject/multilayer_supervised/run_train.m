% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

%% Setup environment
%  Add common directory to your path for minFunc
addpath ../common;
addpath(genpath('../common/minFunc_2012/minFunc'));
addpath(genpath('.'));
addpath(genpath('../preprocessing'));

  
%% Load the MNIST Data
[train,test] = ex1_load_mnist();

% Add row of 1s to the dataset to act as an intercept term.
train.y = train.y+1; % make labels 1-based.
test.y  = test.y+1; % make labels 1-based.

train_X = train.X;
train_y = train.y;

test_X  = test.X;
test_y  = test.y;

% Training set info
m = size(train.X,2);
n = size(train.X,1);


%% Network hyperparameters
% ei is a structure you can use to store hyperparameters of the network
ei = [];
ei.input_dim = n;
ei.output_dim = 10;
ei.layer_sizes = [30, 20, ei.output_dim];
ei.lambda = 1e-6;
ei.activation_fun = 'logistic';
%ei.activation_fun = 'tanh';
%ei.activation_fun = 'relu';


%% Setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% Setup minFunc options
options = [];
options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';
options.maxIter = 50;

%% Training with minFunc
b_min = cputime;
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, train_X, train_y);
e_min = cputime - b_min;
disp(e_min);

%% Training with stochastic gradient descent
alpha = 0.01;
b_sgd = cputime;
[opt_params, error] = stochastic_grad_desc(@supervised_dnn_cost, params, alpha, 100, train_X, train_y, test_X, test_y, ei); 
e_sgd = cputime - b_sgd;
disp(e_sgd);

%% Accuracy on test and train set.
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, test_X, [], true);
[~,pred] = max(pred);
acc_test = mean(pred==test_y);
fprintf('test accuracy: %f\n', acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, train_X, [], true);
[~,pred] = max(pred);
acc_train = mean(pred==train_y);
fprintf('train accuracy: %f\n', acc_train);





