
%% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

% setup environmentc
% experiment information
% a struct containing network layer sizes etc
ei = [];

%Makes code easier to read
NIMSTIM=1;
POFA=2;

% add common directory to your path for
% minfunc and mnist data helpers
addpath(pwd); 
addpath(fullfile('..', 'common'));
addpath(genpath(fullfile('..', 'common','minFunc_2012','minFunc')));
addpath(genpath(fullfile('..', 'common','minFunc_2012','autoDif')));
addpath(genpath(fullfile('..', 'common', 'Mohammad Haghighat')));  %From Matlab exchange

% TODO: load face data
% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)

%Pre-processing pipeline: includes loading data and instantiating network
p3;

%% Train with minFunc
for i=1:length(network)
    % setup minfunc options
    options.display = 'iter';
    options.maxFunEvals = 1e6;
    options.Method = 'lbfgs';
    ei = network(i).network_design.ei;
    params = stack2params(initialize_weights(ei)); 
    data=network(i).network_input.features;
    label=network(i).network_input.labels;
    pred_only=0; 
    
    % run training
   
    [opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost, params,  options, ei, data, label, pred_only);  
    
    % TODO:  1) check the gradient calculated by supervised_dnn_cost.m
    %        2) Decide proper hyperparamters and train the network.
    %        3) Implement SGD version of solution.
    %        4) Plot speed of convergence for 1 and 3.
    %        5) Compute training time and accuracy of train & test data.
    
    % compute accuracy on the test and train set
    [~, ~, pred] = supervised_dnn_cost( opt_params, ei{i}, data_test, [], true);
    [~,pred] = max(pred);
    acc_test = mean(pred'==labels_test);
    fprintf('test accuracy: %f\n', acc_test);
    
    [~, ~, pred] = supervised_dnn_cost( opt_params, ei{i}, data_train, [], true);
    [~,pred] = max(pred);
    acc_train = mean(pred'==labels_train);
    fprintf('train accuracy: %f\n', acc_train);
end

%% Train with gradientdescent; leave 2 out line search for lambda (L2 regularization);
    % setup minfunc options
    options = [];
    options.display = 'iter';
    options.maxFunEvals = 1e6;
    options.Method = 'lbfgs';
    ei = network(i).network_design.ei;
    params = stack2params(initialize_weights(ei)); 
    data=network(i).network_input.features;
    label=network(i).network_input.labels;
    
% LIAM:  Code fails here
% net_input = {NetworkInput(ei{NIMSTIM}, NIMSTIM)};
% for i=1:length(ei)
%     for j=1:1
%         stack = initialize_weights(ei{i});
%         params = stack2params(stack);
%         
%         % setup minfunc options
%         options = [];
%         options.display = 'iter';
%         options.maxFunEvals = 1e6;
%         options.Method = 'lbfgs';
%         
%         [opt_params,opt_value,exitflag,output] = gradientdescent( dnn_cost_function, params, options, ei, data, labels);
%         
%         %Makes code easier to read
%         NimStim=1;
%         POFA=2;
%         
%         inputs = cellfun(@(x) NetworkInput(x.ei, x.data), networkintput_input);
%         network = cellfun(@(x) Network(x), inputs);
%         outputs = cellfun(@(x) NetworkOutput(x), network);
%     end
% end






