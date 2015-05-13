dataset_locs={fullfile('..','data','NimStim'), fullfile('..','data','POFA')}; %populate cell array of photo directories
paths = cellfun(@(x) dir(x), dataset_locs, 'UniformOutput', false)'; %populate cell array of photo file paths

%Key values for the two data sets
NIMSTIM=1;
POFA=2;

%Load data into fulldata cellarray and target values into target
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
    target{i}=cellfun( @(x) getImageLabels(x, i), filename{i});
end

%% Scale images to the proper size
processed_images.NimStim.images = cellfun(@(x) imresize(rgb2gray(x),[64,64]), fulldata{1}.data, 'UniformOutput', false)
processed_images.NimStim.targets = target{1}
processed_images.POFA.images = cellfun(@(x) imresize(x,[64,64]), fulldata{2}.data, 'UniformOutput', false)
processed_images.POFA.targets = target{2}

%% Generate Gabor Filters

%add gabor filter library to path:
addpath(genpath(fullfile('..', 'preprocessing', 'gabor')));  %From Matlab exchange

ei.orientations = 8;
ei.scales = 5;
ei.scaleDownFactor = 8;

gabArr = gaborFilterBank(ei.orientations, ei.scales, ei.scaleDownFactor, ei.scaleDownFactor);

%% Preprocess images to gabor filters
processed_images.NimStim.images = cellfun(@(x) gaborFeatures(x, gabArr,8, 8) , processed_images.NimStim.images, 'UniformOutput' , false);
processed_images.POFA.images = cellfun(@(x) gaborFeatures(x, gabArr,8, 8) , processed_images.POFA.images, 'UniformOutput' , false);


