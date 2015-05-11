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
        function obj = Network(design, full_data)
            %Check input
            assert(isa(design, 'NetworkDesign'), 'Network (constructor): design must be of NetworkDesign class');
            assert(isa(input, 'NetworkInput'), 'Network (constructor): design must be of NetworkDesign class');
            assert(length(input)==length(design),'Network (constructor): input and design must be same length');
            
            %Create a single network for each input/design pair
            if length(input)==1 && length(design)==1
                obj.network_design=design;
                obj.train_input=train_input;
                obj.test_input=test_input;
                obj.network_output = NetworkOutput(obj);
                return;
                
                %We are passed a vector of inputs/designs: recursively call
                %for each pair.
            else
                n=1;
                for i=1:length(design)
                    obj(n) = Network(design(i), train_input(i), test_input(i));
                    n=n+1;
                end
            end
        end
        
        %Train the network on train_input
        function train(obj)
            for i=1:length(obj) 
                %Iterate until MAXITER or converged
                n=1; 
                while obj(i).checkConvergence || (n > obj.MAXITER)
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
        
    end
    
    methods(Hidden=true)
     
        %Use gradient descent to learn weights for the current network activation 
        function backProp(obj)
        end
        
        %Compute all network activations
        function forwardProp(obj)
            a{i}
            for i=1:length(obj.network_output.stack)
                this_weight = obj.network_output.stack{i};
                a{i}=tanh(this_weight.W*obj.network_output.deltas{i};
            end
            
            
        end
        
        %2.b - Check gradient using numerical approximations (unit test for gradient)
        function checkGradient(obj)
            return;
        end
        
        function checkConverged(obj)
            return;
        end
        
        function loss(obj)
            return;
        end
        
    end
end

