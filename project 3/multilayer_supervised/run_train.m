%% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

% setup environment
% experiment information
% a struct containing network layer sizes etc
ei = [];

%Makes code easier to read
NIMSTIM=1;
POFA=2;
    
% add common directory to your path for
% minfunc and mnist data helpers
addpath(fullfile('..', 'common'));
addpath(genpath(fullfile('..', 'common','minFunc_2012','minFunc')));

% TODO: load face data
dataset_locs={fullfile('..','data','NimStim'), fullfile('..','data','POFA')};
paths = cellfun(@(x) dir(x), dataset_locs, 'UniformOutput', false)';

%Loop over two datasets
for i=1:length(paths)
    %Load NimStim as cell array of matrices. We need a dummy 'ErrorHandler' to to tell it to pass if nothing was loaded
    file_paths{i} = fullfile(dataset_locs{i}, {paths{i}(:).name});
    data{i} = cellfun(@(x) imread(x), file_paths{i}, 'UniformOutput', false, 'ErrorHandler', @(x,y) 0)';
    good_idx{i} = cellfun(@(x) nnz(x) > 0, data{i}, 'ErrorHandler', @(x,y) 0);
    data{i} = data{i}(good_idx{i});
    
    %Save filename to extract label
    temp = regexp({paths{i}(good_idx{i}).name}, '[\\\/.]', 'split')';
    filename{i} = cellfun(@(x) x{1}, temp, 'UniformOutput', false);
    label{i}=cellfun(@NetworkInput.filename2Label, filename{i});
end

%Get unique states
unique_nstim_state=unique(upper({label{NIMSTIM}(:).state}))'; 
unique_pofa_state=unique({label{POFA}(:).state})'; 
unique_state={unique_nstim_state, unique_pofa_state}; 

%Get unique ids
unique_nstim_id = unique(upper({label{NIMSTIM}(:).id}))';
unique_pofa_id=unique({label{POFA}(:).id})'; 
unique_id={unique_nstim_id, unique_pofa_id}; 

% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)

% NimStim:
ei(NIMSTIM).input_dim = length(data{NIMSTIM})*40; 
ei(NIMSTIM).output_dim = length(unique_id{NIMSTIM});
% sizes of all hidden layers and the output layer
ei(NIMSTIM).layer_sizes = [ceil(ei(NIMSTIM).input_dim/ei(NIMSTIM).output_dim), ei(NIMSTIM).output_dim];
% scaling parameter for l2 weight regularization penalty
ei(NIMSTIM).lambda = 1;
% which type of activation function to use in hidden layers
ei(NIMSTIM).activation_fun = 'logistic';
%ei.activation_fun = 'tanh';

% POFA: 
ei(POFA).input_dim = length(data{POFA})*40; 
ei(POFA).output_dim = length(unique_state{POFA});
% sizes of all hidden layers and the output layer
ei(POFA).layer_sizes = [10, ei(POFA).output_dim];
% scaling parameter for l2 weight regularization penalty
ei(POFA).lambda = 1;
% which type of activation function to use in hidden layers
ei(POFA).activation_fun = 'logistic';
%ei.activation_fun = 'tanh';


%% Train with minFunc
for i=1:length(ei)
    stack = initialize_weights(ei(i));
    params = stack2params(stack);
    
    % setup minfunc options
    options = [];
    options.display = 'iter';
    options.maxFunEvals = 1e6;
    options.Method = 'lbfgs';
    
    % run training
    [opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
        params,options,ei(i), data_train, labels_train);
    
    % TODO:  1) check the gradient calculated by supervised_dnn_cost.m
    %        2) Decide proper hyperparamters and train the network.
    %        3) Implement SGD version of solution.
    %        4) Plot speed of convergence for 1 and 3.
    %        5) Compute training time and accuracy of train & test data.
    
    % compute accuracy on the test and train set
    [~, ~, pred] = supervised_dnn_cost( opt_params, ei(i), data_test, [], true);
    [~,pred] = max(pred);
    acc_test = mean(pred'==labels_test);
    fprintf('test accuracy: %f\n', acc_test);
    
    [~, ~, pred] = supervised_dnn_cost( opt_params, ei(i), data_train, [], true);
    [~,pred] = max(pred);
    acc_train = mean(pred'==labels_train);
    fprintf('train accuracy: %f\n', acc_train);
end

%% Train with gradientdescent; leave 2 out line search for lambda (L2 regularization);
% LIAM:  Code fails here
net_input = {NetworkInput(ei{NIMSTIM}, NIMSTIM)};
for i=1:length(ei)
    for j=1:1
        stack = initialize_weights(ei(i));
        params = stack2params(stack);

        % setup minfunc options
        options = [];
        options.display = 'iter';
        options.maxFunEvals = 1e6;
        options.Method = 'lbfgs';

        [opt_params,opt_value,exitflag,output] = gradientdescent( dnn_cost_function, params, options, ei, data, labels);

        %Makes code easier to read
        NimStim=1;
        POFA=2;

        inputs = cellfun(@(x) NetworkInput(x.ei, x.data), networkintput_input);
        network = cellfun(@(x) Network(x), inputs);
        outputs = cellfun(@(x) NetworkOutput(x), network);   
    end
end
    
    
    
    
    
    
