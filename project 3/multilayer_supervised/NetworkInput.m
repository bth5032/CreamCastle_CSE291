classdef NetworkInput < matlab.mixin.Copyable
    %NETWORKINPUT Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        ei
        data
        convolved_features
        
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
        function obj = NetworkInput(ei, data)
            obj.ei   = ei;
            obj.data = data;
        end
        
        
        function preprocess_all(obj)
            obj.convolved_features = cellfun(@(x) preprocess(obj,x), obj.data, 'UniformOutput', false);
        end
        
       
        function convolved_feature = preprocess(obj, img)
            % Input
            %        img   :    The RGB image in matrix form to be preprocessed 
            %
            % Output
            %        convolved_feature : The flattened array of 40 
            %                            convolved images rescaled by a 
            %                            factor of 1/8 for a single image
            
            % Resize a single image and convert image to gray-scale
            temp   = rgb2gray(imresize(img, [64, 64]));
            gabArr = obj.gaborFilterBank(5, 8, 96, 96);
            convolved_feature = obj.gaborFeatures(temp, gabArr, 8, 8);
        end
        
        
        function gaborArray = gaborFilterBank(obj,u,v,m,n)
            
            % GABORFILTERBANK generates a custom Gabor filter bank.
            % It creates a u by v array, whose elements are m by n matries;
            % each matrix being a 2-D Gabor filter.
            %
            %
            % Inputs:
            %       u	:	No. of scales (usually set to 5)
            %       v	:	No. of orientations (usually set to 8)
            %       m	:	No. of rows in a 2-D Gabor filter (an odd integer number usually set to 39)
            %       n	:	No. of columns in a 2-D Gabor filter (an odd integer number usually set to 39)
            %
            % Output:
            %       gaborArray: A u by v array, element of which are m by n
            %                   matries; each matrix being a 2-D Gabor filter
            %
            %
            % Sample use:
            %
            % gaborArray = gaborFilterBank(5,8,39,39);
            %
            %
            %   Details can be found in:
            %
            %   M. Haghighat, S. Zonouz, M. Abdel-Mottaleb, "Identification Using
            %   Encrypted Biometrics," Computer Analysis of Images and Patterns,
            %   Springer Berlin Heidelberg, pp. 440-448, 2013.
            %
            %
            % (C)	Mohammad Haghighat, University of Miami
            %       haghighat@ieee.org
            %       I WILL APPRECIATE IF YOU CITE OUR PAPER IN YOUR WORK.
            if (nargin ~= 5)    % Check correct number of arguments
                error('There should be four inputs.')
            end
            
            %% Create Gabor filters
            
            % Create u*v gabor filters each being an m*n matrix
            gaborArray = cell(u,v);
            fmax = 0.25;
            gama = sqrt(2);
            eta = sqrt(2);
            
            for i = 1:u
                
                fu = fmax/((sqrt(2))^(i-1));
                alpha = fu/gama;
                beta = fu/eta;
                
                for j = 1:v
                    tetav = ((j-1)/v)*pi;
                    gFilter = zeros(m,n);
                    
                    %Other data, labels, etc.
                    obj(i).unique_states=fulldata{i}.unique_state;
                    obj(i).unique_ids=fulldata{i}.unique_id;
                    
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
                    
                    %Make cross-validation folds
                    obj.folds = obj.makeXvalFolds(fulldata);
                end
            end
        end
        
        function featureVector = gaborFeatures(obj, img, gaborArray, d1, d2)
            
            % GABORFEATURES extracts the Gabor features of the image.
            % It creates a column vector, consisting of the image's Gabor features.
            % The feature vectors are normalized to zero mean and unit variance.
            %
            %
            % Inputs:
            %       img         :	Matrix of the input image
            %       gaborArray	:	Gabor filters bank created by the function gaborFilterBank
            %       d1          :	The number of rows of the output image.
            %       d2          :	The number of columns of the output image.
            %
            % Output:
            %       featureVector	:   A column vector with length (m*n*u*v)/(d1*d2).
            %                           This vector is the Gabor feature vector of an
            %                           m by n image. u is the number of scales and
            %                           v is the number of orientations in 'gaborArray'.
            %
            %
            % Sample use:
            %
            % img = imread('cameraman.tif');
            % gaborArray = gaborFilterBank(5,8,39,39);  % Generates the Gabor filter bank
            % featureVector = gaborFeatures(img,gaborArray,8, 8);   % Extracts Gabor feature vector, 'featureVector', from the image, 'img'.
            %
            %
            %   Details can be found in:
            %
            %   M. Haghighat, S. Zonouz, M. Abdel-Mottaleb, "Identification Using
            %   Encrypted Biometrics," Computer Analysis of Images and Patterns,
            %   Springer Berlin Heidelberg, pp. 440-448, 2013.
            %
            %
            % (C)	Mohammad Haghighat, University of Miami
            %       haghighat@ieee.org
            %       I WILL APPRECIATE IF YOU CITE OUR PAPER IN YOUR WORK.
            
            if (nargin ~= 5)    % Check correct number of arguments
                error('Use correct number of input arguments!')
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
                
                %TODO: populate obj.params_participant_map in Network
                %constructor
                
                %TODO: take a single fulldata struct and split into train
                %and test structs based on participants (with the same fields as fulldata)
                train_data=[];
                test_data=[]; 
                
                %Create NetworkXvalFold object
                %folds=NetworkXvalFold(NetworkInput(train_data), NetworkInput(test_data));
            end
            
            
            %% Feature Extraction
            
            % Extract feature vector from input image
            [n,m] = size(img);
            s = (n*m)/(d1*d2);
            l = s*u*v;
            featureVector = zeros(l,1);
            c = 0;
            for i = 1:u
                for j = 1:v
                    
                    c = c+1;
                    gaborAbs = abs(gaborResult{i,j});
                    gaborAbs = imresize(gaborAbs, [d1,d2]);
                    %piazza suggested imresize instead of downsample
                    %gaborAbs = downsample(gaborAbs,d1);
                    %gaborAbs = downsample(gaborAbs.',d2);
                    gaborAbs = reshape(gaborAbs.',[],1);
                    
                    % Normalized to zero mean and unit variance. (if not applicable, please comment this line)
                    gaborAbs = (gaborAbs-mean(gaborAbs))/std(gaborAbs,1);
                    
                    featureVector(((c-1)*s+1):(c*s)) = gaborAbs;
                    
                end
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

