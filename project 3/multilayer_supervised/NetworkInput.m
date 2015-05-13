classdef NetworkInput < matlab.mixin.Copyable
    %NETWORKINPUT Prepares raw data for input to deep belief net
    
    properties
        features
        labels
        
        xval_map
        
        folds
    end
    
    properties(Constant)
        IMAGEDIM=64;
        FEATUREDIM=8;
        FILTERDIM=96;
        ORIENTATIONS=5;
        SCALES=8;
        NUM_COMPONENTS=40;
        NUMFOLDS=5;
    end
    
    methods
        
        %Constructor: fulldata can be a cell array of dataset structs
        function obj = NetworkInput(fulldata)
            
            %Generate filters; 5 orientations, 8 scales
            gabArr = gaborFilterBank(obj.ORIENTATIONS, obj.SCALES, obj.FILTERDIM, obj.FILTERDIM);
            
            for i=1:length(fulldata)
                %Given a cell array of datasets
                if iscell(fulldata)
                    obj(i) = NetworkInput(fulldata{i});
                    continue;
                    
                else
                    %Resize
                    all_images = cellfun(@(x) imresize(x, [obj.IMAGEDIM, obj.IMAGEDIM]),...
                        fulldata.data, 'uniformoutput',false);
                    
                    %Filter and downsample images
                    all_512_gabor_features = cellfun(@(x) gaborFeatures(x, gabArr,...
                        [obj.FEATUREDIM, obj.FEATUREDIM]), all_images, 'UniformOutput' , false);
                    
                    %PCA each scale of each image, reducing dimension across
                    %pixels and orientation.
                    temp = cellfun(@(x) obj.getCellFeatures(x), all_512_gabor_features, 'UniformOutput', false);
                    
                    %Concatenate features across images
                    stacked_gabor_features=cell2mat(temp);
                    
                    %Normalize/vectorize features across images and scales
                    scored_gabor_features = zscore(zscore(stacked_gabor_features)');
                    
                    %PCA/zscore: Normalize top-40 PCs. Save as obj.features
                    obj.features = scored_gabor_features(:);
                    
                    %TODO: populate labels with integer class identifiers
                    %instead of strings
                    %obj.labels=fulldata.labels;
                    
                    %TODO: populate obj.xval_map mapping from
                    %cross-validation folds to features
                    %all_tags = fulldata.(fulldata.xval_tag);
                    %unique_tags=unique(all_tags)
                    
                    %TODO: Make cross-validation folds
                    %obj.folds = obj.makeXvalFolds(fulldata);
                end
            end
        end
        
        %Pick a scale: you have 8 orientations, so 8*(8*8) variables, PCA
        %on this and pick top 8. Repeat for each scale: 8*5 = 40 dimensions.
        function mat = getCellFeatures(obj, cell)
            %Cycle through 5 scales per image, PCA each scale
            mat=[];
            for i=1:size(cell, 1)
                temp=cell2mat(cell(i,:));
                [U,S] = svds(zscore(temp), obj.NUM_COMPONENTS);
                fvec=U*S;
                mat=[mat fvec(:)];
            end
        end
        
        %Take fulldata into xval folds
        function folds = makeXvalFolds(obj, fulldata, fold_num)
            if ~exist('fold_num','var')
                fold_num=0;
            end
            
            if iscell(fulldata)
                %Create fold for each dataset
                for i=1:num_folds
                    for j=1:length(fulldata)
                        folds(i,j) = NetworkInput.makeXvalFolds(fulldata{j}, num_folds, i);
                    end
                end
            else
                
                obj.NUMFOLDS;
                
                %TODO: populate obj.xval_map in Network constructor
                
                %TODO: take a single fulldata struct and split into train
                %and test structs based on participants (with the same fields as fulldata)
                
                train_data=[];
                test_data=[];
                
                %Create NetworkXvalFold object
                folds=NetworkXvalFold(NetworkInput(train_data), NetworkInput(test_data));
            end
        end
    end
    
    methods(Static=true)
        %RGB or grayscales can come through
        function gray_image = toGray(image)
            if size(image, 3) > 1
                gray_image=rgb2gray(image);
            else
                gray_image=image;
            end
        end
        
        %Parse the filename into label struct
        function label = filename2Label(filename)
            temp=regexp(filename,'[_-]','split');
            label.id=temp{1}(1:2);
            label.state=temp{2};
        end
        
    end
end

