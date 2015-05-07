classdef NetworkInput < matlab.mixin.Copyable
    %NETWORKINPUT Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        ei
        data
        convolved_features
        
        %From stack2params(stack)
        params
        params_participant_map
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
                    
                    for x = 1:m
                        for y = 1:n
                            xprime = (x-((m+1)/2))*cos(tetav)+(y-((n+1)/2))*sin(tetav);
                            yprime = -(x-((m+1)/2))*sin(tetav)+(y-((n+1)/2))*cos(tetav);
                            gFilter(x,y) = (fu^2/(pi*gama*eta))*exp(-((alpha^2)*(xprime^2)+(beta^2)*(yprime^2)))*exp(1i*2*pi*fu*xprime);
                        end
                    end
                    gaborArray{i,j} = gFilter;
                    
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
            
            if size(img,3) == 3	% % Check if the input image is grayscale
                img = rgb2gray(img);
            end
            
            img = double(img);
            
            
            %% Filtering
            
            % Filter input image by each Gabor filter
            [u,v] = size(gaborArray);
            gaborResult = cell(u,v);
            for i = 1:u
                for j = 1:v
                    gaborResult{i,j} = conv2(img,gaborArray{i,j},'same');
                    % J{u,v} = filter2(G{u,v},I);
                end
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
       
       %Parse the NimStim filename into label struct
       function label = filename2Label(filename)
           temp=regexp(filename,'[_-]','split');
           label.id=temp{1}(1:2);
           label.state=temp{2};
       end
   end
   
end

