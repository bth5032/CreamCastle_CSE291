classdef NetworkInput < matlab.mixin.Copyable
    %NETWORKINPUT Prepares raw data for input to deep belief net
    
    properties
        data
        features
        
        unique_states
        unique_ids
        
        %From stack2params(stack)
        params
        params_participant_map
    end
    
    methods
        
        
        %Constructor: fulldata can be a cell array of datasets
        %fulldata.data (images)
        %fulldata.unique_state (output state-space)
        %fulldata.unique_id (unique identifier of the participant)
        function obj = NetworkInput(fulldata)
            IMAGEDIM=64;
            FEATUREDIM=8;
            FILTERDIM=96;
            ORIENTATIONS=5;
            SCALES=8;
            NUM_COMPONENTS=40;
            
            for i=1:length(fulldata)
                
                %Given a cell array of datasets
                if iscell(fulldata)
                    obj(i) = NetworkInput(fulldata{i});
                    continue;
                else
                    this_item=fulldata(i);
                end
                
                temp = cellfun(@(x) obj.toGray(imresize(x, [IMAGEDIM, IMAGEDIM])), this_item.data,'UniformOutput' , false);
                
                %Generate filters; 5 orientations, 8 scales
                gabArr = gaborFilterBank(ORIENTATIONS, SCALES, FILTERDIM, FILTERDIM);
                
                %Filter and downsample images
                all_512_gabor_features = cellfun(@(x) gaborFeatures(x, gabArr,...
                    [FEATUREDIM, FEATUREDIM]), temp, 'UniformOutput' , false);
                
                %Configure the normalized features as a tall matrix (N x D),
                %where N = #{data samples} and D = #{features}
                temp = cellfun(@(x) obj(i).vectorizeCell(x), all_512_gabor_features, 'UniformOutput', false);
                stacked_gabor_features=cell2mat(temp')';
                
                %Normalize features
                scored_gabor_features = zscore(stacked_gabor_features);
                
                %Take top NUM_COMPONENTS singular values and principle vectors.
                [U,S] = svds(scored_gabor_features, NUM_COMPONENTS);
                
                %PCA/zscore: Normalize top-40 PCs. Save as obj.features
                obj(i).features = zscore(U*S);
                
                %Other data, labels, etc.
                obj(i).unique_states=fulldata.unique_state{i};
                obj(i).unique_ids=fulldata.unique_id{i};
            end
        end
        
        function vec = vectorizeCell(obj, cell)
            mat=cell2mat(cell);
            vec=mat(:);
        end
        
        %RGB or grayscales can come through
        function gray_image = toGray(obj, image)
            if size(image, 3) > 1
                gray_image=rgb2gray(image);
            else
                gray_image=image;
            end
        end
    end
    
    methods(Static=true)
        %Parse the filename into label struct
        function label = filename2Label(filename)
            temp=regexp(filename,'[_-]','split');
            label.id=temp{1}(1:2);
            label.state=temp{2};
        end
        
        function folds = makeXvalFolds(fulldata, num_folds)
            N=ceil(length(fulldata)/num_folds);
            for i=1:num_folds
                start_idx = (i-1)*N+1;
                %Don't exceed the dimension of the data vector; last chunk
                %can be short
                end_idx = start_idx:min(start_idx+N, length(fulldata.data))
                this_range = 1:();
            end
            
        end
        
        
    end
    
