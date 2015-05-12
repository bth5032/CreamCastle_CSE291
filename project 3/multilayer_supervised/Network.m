classdef Network < matlab.mixin.Copyable
    %NETWORK
    
    properties
        network_design
        network_input
        network_output
    end
    
    properties(Constant)
        MAXITER=1000;
    end
    
    methods
        
        %Constructor: takes NetworkDesign and NetworkInput ojbects as
        %arguments.
        function obj = Network(network_design, network_input)
            %Check input
            assert(isa(network_design, 'NetworkDesign'), 'Network (constructor): design must be of NetworkDesign class');
            assert(isa(network_input, 'NetworkInput'), 'Network (constructor): design must be of NetworkDesign class');
            assert(length(network_input)==length(network_design),'Network (constructor): input and design must be same length');
            
            %Create a single network for each input/design pair
            if length(network_design)==1
                obj.network_design=copy(network_design);
                obj.network_input=copy(network_input);
                obj.network_output = NetworkOutput(obj); %NetworkOutput keeps a backlink to this object
                return;
                
                %We are passed a vector of inputs/designs: recursively call
                %for each pair.
            else
                for i=1:length(network_design)
                    obj(i) = Network(network_design(i), network_input(i));
                end
            end
        end
        
        %Train the network on train_input
        function train(obj)
            for i=1:length(obj) 
                %Iterate until MAXITER or converged
                n=1; 
                while obj(i).checkConvergence || ~(n > obj.MAXITER)
                    fprintf('Network.train: training Network %d/%d\n', i, length(obj));
                    folds = obj(i).network_input.makeXvalFolds;
                    obj(i).forwardProp;
                    obj(i).backProp(folds);
                    n=n+1; 
                end
            end
        end
        
        %Test the network on test_input
        function test(obj)
        end 
        
        %Make Network objects serializable
        function saveObj(obj)
        end
        
    end
    
    methods(Hidden=true)
        % Use gradient descent to learn weights for the current network activation 
        function backProp(obj)
            
        end
        
        %Compute all network activations
        function forwardProp(obj)
            act=obj.network_input.features; 
            afunc=obj.network_design.activationFunction;
            
            obj.network_output.stack
            for i=1:length(obj.network_output.stack)
                this_weight = obj.network_output.stack{i};
                act = afunc(this_weight.W*act + this_weight.b);
                obj.network_output.activations{i}=act;
            end    
        end
        
        %2.b - Check gradient using numerical approximations (unit test for gradient)
        function checkGradient(obj)
            return;
        end
        
        function checkConverged(obj)
            return;
        end
       
        % Cross-entropy loss for neural network
        function loss = lossFunc(obj)
            loss = 0;
            
            % Index of the hidden layer directly before output layer
            layer_index = length(obj.network_output.activations)-1;
            
            % Calc denominator of cross-entropy loss
            denominator = 0;
            for j=1:length(obj.network_design.ei.output_dim)
                denominator = denominator + dot(obj.network_output.activations{layer_index}, obj.network_output.stack{layer_index}(j));
            end
            
            for i=1:length(fulldata.data)
                for k=1:length(obj.network_design.ei.output_dim)
                    numerator   = dot(obj.network_output.activations{layer_index}, obj.network_output.stack{layer_index}(k));
                    loss = loss -1*(obj.network_input.labels==k)*log(numerator/denominator);
                end
            end
        end
        
    end
    
    methods(Static)
        %Make Network objects serializable
        function loadObj
        end
    end
end

