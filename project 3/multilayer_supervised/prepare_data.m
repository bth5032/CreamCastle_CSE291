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

gabArr = gaborFilterBank(ei.scales, ei.orientations, ei.scaleDownFactor, ei.scaleDownFactor);

%% Preprocess images to gabor filters
feature_vector.NimStim.images = cellfun(@(x) gaborFeatures(x, gabArr,8, 8) , processed_images.NimStim.images, 'UniformOutput' , false);
feature_vector.NimStim.targets = target{1}
feature_vector.POFA.images = cellfun(@(x) gaborFeatures(x, gabArr,8, 8) , processed_images.POFA.images, 'UniformOutput' , false);
feature_vector.POFA.targets = target{2}
%% make feature vector 3D matrix
temp_features.NimStim = feature_vector.NimStim.images{1}
for i=2:length(feature_vector.NimStim.images)
    temp_features.NimStim(:,:,i) = feature_vector.NimStim.images{i}
end
temp_features.POFA = feature_vector.NimStim.images{1}
for i=2:length(feature_vector.POFA.images)
    temp_features.POFA(:,:,i) = feature_vector.POFA.images{i}
end

%% split matricies by scale
temp2_features.NimStim.scale1 = squeeze(temp_features.NimStim(:,1,:));
temp2_features.NimStim.scale2 = squeeze(temp_features.NimStim(:,2,:));
temp2_features.NimStim.scale3 = squeeze(temp_features.NimStim(:,3,:));
temp2_features.NimStim.scale4 = squeeze(temp_features.NimStim(:,4,:));
temp2_features.NimStim.scale5 = squeeze(temp_features.NimStim(:,5,:));

temp2_features.POFA.scale1 = squeeze(temp_features.POFA(:,1,:));
temp2_features.POFA.scale2 = squeeze(temp_features.POFA(:,2,:));
temp2_features.POFA.scale3 = squeeze(temp_features.POFA(:,3,:));
temp2_features.POFA.scale4 = squeeze(temp_features.POFA(:,4,:));
temp2_features.POFA.scale5 = squeeze(temp_features.POFA(:,5,:));

%% Find SVD projectors 

[U,S,V] = svds(temp2_features.NimStim.scale1', 8);
NimStim.p1 = V';
[U,S,V] = svds(temp2_features.NimStim.scale2', 8);
NimStim.p2 = V';
[U,S,V] = svds(temp2_features.NimStim.scale3', 8);
NimStim.p3 = V';
[U,S,V] = svds(temp2_features.NimStim.scale4', 8);
NimStim.p4 = V';
[U,S,V] = svds(temp2_features.NimStim.scale5', 8);
NimStim.p5 = V';

[U,S,V] = svds(temp2_features.POFA.scale1', 8);
POFA.p1 = V';
[U,S,V] = svds(temp2_features.POFA.scale2', 8);
POFA.p2 = V';
[U,S,V] = svds(temp2_features.POFA.scale3', 8);
POFA.p3 = V';
[U,S,V] = svds(temp2_features.POFA.scale4', 8);
POFA.p4 = V';
[U,S,V] = svds(temp2_features.POFA.scale5', 8);
POFA.p5 = V';

%% Project Feature Vectors to 8 dimensions
Final_NimStim_Input_Matrix = vertcat(NimStim.p1*temp2_features.NimStim.scale1, NimStim.p2*temp2_features.NimStim.scale2, ...
    NimStim.p3*temp2_features.NimStim.scale3, NimStim.p4*temp2_features.NimStim.scale4, NimStim.p5*temp2_features.NimStim.scale5)

Final_POFA_Input_Matrix = vertcat(POFA.p1*temp2_features.POFA.scale1, POFA.p2*temp2_features.POFA.scale2, ...
    POFA.p3*temp2_features.POFA.scale3, POFA.p4*temp2_features.POFA.scale4, POFA.p5*temp2_features.POFA.scale5)

%% Write out targets
Final_NimStim_Targets = target{1}'

Final_POFA_Targets = target{2}'

