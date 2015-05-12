%% Get data from disk
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

%% Create NetworkInput object

%Get states
fulldata{NIMSTIM}.state=upper({label{NIMSTIM}(:).state})'; 
fulldata{POFA}.state={label{POFA}(:).state}'; 

%Get ids
fulldata{NIMSTIM}.id=upper({label{NIMSTIM}(:).id})';
fulldata{POFA}.id={label{POFA}(:).id}'; 

%Get labels
fulldata{NIMSTIM}.labels = fulldata{NIMSTIM}.id;
fulldata{POFA}.labels = fulldata{POFA}.state;

%Select cross-validation 
fulldata{NIMSTIM}.xval_tag='state';
fulldata{POFA}.xval_tag='id';

network_input = NetworkInput(fulldata);

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
%ei{NIMSTIM}.activation_fun = 'tanh';


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
%ei{POFA}.activation_fun = 'tanh';

network_design = NetworkDesign(ei); 

%% Create Network object
network=Network(network_design, network_input);






