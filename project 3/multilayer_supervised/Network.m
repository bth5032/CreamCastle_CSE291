classdef Network < matlab.mixin.Copyable
    %NETWORK
    
    properties
        network_design
        train_input
        test_input
        
        network_output
        network_converged=0
    end
    
    methods
        
        %Constructor: takes NetworkDesign and NetworkInput ojbects as
        %arguments.
        function obj = Network(design, train_input, test_input)
            %Check input
            assert(isa(design, 'NetworkDesign'), 'Network (constructor): design must be of NetworkDesign class');
            assert(isa(input, 'NetworkInput'), 'Network (constructor): design must be of NetworkDesign class');
            assert(length(input)==length(design),'Network (constructor): input and design must be same length'); 
            
            %Create a single network for each input/design pair 
            if length(input)==1 && length(design)==1
                obj.network_design=design;
                obj.train_input=train_input;
                obj.test_input=test_input;
                obj.network_output = NetworkOutput;
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
        end
        
        %Test the network on test_input
        function test(obj)
        end
        
        %Compute all network activations 
        function forwardProp(obj)
            obj.network_design.ei.inputdim
        end

    end
    
    methods(Hidden=true)
        %Make one gradient descent step in this network
        function stepLearnWeights(obj, num_iterations)
        end
        
        %Perform backprop on network
        function gradientDesc(obj)
            return;
        end
        
        %2.b - Check gradient using numerical approximations (unit test for gradient)
        function checkGradient(obj)
            return;
        end
        
        function loss(obj)
            return;
        end
        
    end
end

