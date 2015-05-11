%% Get data from disk, Create NetworkInput object
dataset_locs={fullfile('..','data','NimStim'), fullfile('..','data','POFA')};
paths = cellfun(@(x) dir(x), dataset_locs, 'UniformOutput', false)';

%Makes code easier to read
NIMSTIM=1;
POFA=2;

%Loop over two datasets
for i=1:length(paths)
    %Load NimStim as cell array of matrices. We need a dummy 'ErrorHandler' to to tell it to pass if nothing was loaded
    file_paths{i} = fullfile(dataset_locs{i}, {paths{i}(:).name});
    fulldata{i}.data = cellfun(@(x) imread(x), file_paths{i}, 'UniformOutput', false, 'ErrorHandler', @(x,y) 0)';
    good_idx{i} = cellfun(@(x) nnz(x) > 0, fulldata{i}.data, 'ErrorHandler', @(x,y) 0);
    fulldata{i}.data = fulldata{i}.data(good_idx{i});
    
    %Remove to get full data, not just top 10
    fulldata{i}.data=fulldata{i}.data(1:10); 
    
    %Save filename to extract label
    temp = regexp({paths{i}(good_idx{i}).name}, '[\\\/.]', 'split')';
    filename{i} = cellfun(@(x) x{1}, temp, 'UniformOutput', false);
    label{i}=cellfun( @NetworkInput.filename2Label, filename{i});
end

%Get unique states
fulldata{NIMSTIM}.unique_state=unique(upper({label{NIMSTIM}(:).state}))'; 
fulldata{POFA}.unique_state=unique({label{POFA}(:).state})'; 
%Get unique ids
fulldata{NIMSTIM}.unique_id=unique(upper({label{NIMSTIM}(:).id}))';
fulldata{POFA}.unique_id=unique({label{POFA}(:).id})'; 


%% Create cross-validation folds (train v. test inputs)
inputs = NetworkInput(fulldata);
%folds = NetworkInput.makeXvalFolds(fulldata, NUMFOLDS);

%% Create NetworkDesign object
% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)

% NimStim:
% dimension of input features FOR YOU TO DECIDE
ei{NIMSTIM}.input_dim = length(fulldata{NIMSTIM}.data)*40; 
% number of output classes FOR YOU TO DECIDE
ei{NIMSTIM}.output_dim = length(fulldata{NIMSTIM}.unique_id);
% sizes of all hidden layers and the output layer
ei{NIMSTIM}.layer_sizes = [ceil(ei{NIMSTIM}.input_dim/ei{NIMSTIM}.output_dim), ei{NIMSTIM}.output_dim];
% scaling parameter for l2 weight regularization penalty
ei{NIMSTIM}.lambda = 1;
% which type of activation function to use in hidden layers
% feel free to implement support for different activation function
ei{NIMSTIM}.activation_fun = 'logistic';
%ei.activation_fun = 'tanh';

% POFA: 
% dimension of input features FOR YOU TO DECIDE
ei{POFA}.input_dim = length(fulldata{POFA}.data)*40; 
% number of output classes FOR YOU TO DECIDE
ei{POFA}.output_dim = length(fulldata{POFA}.unique_state);
% sizes of all hidden layers and the output layer FOR YOU TO DECIDE
ei{POFA}.layer_sizes = [10, ei{POFA}.output_dim];
% scaling parameter for l2 weight regularization penalty
ei{POFA}.lambda = 1;
% which type of activation function to use in hidden layers
% feel free to implement support for different activation function
ei{POFA}.activation_fun = 'logistic';
%ei.activation_fun = 'tanh';

network_design = NetworkDesign(ei); 

%% Create Network
network=Network(network_input, network_design); 

network.train;
network.test;






