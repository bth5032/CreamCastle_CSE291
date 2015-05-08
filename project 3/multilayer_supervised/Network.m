classdef Network < matlab.mixin.Copyable
    %NETWORK
    
    properties
        network_design
        network_input
        network_output
        
        network_converged=0
    end
    
    methods
        
        %Constructor: takes NetworkDesign and NetworkInput ojbects as
        %arguments.
        function obj = Network(input, design)
            %Check input
            assert(isa(design, 'NetworkDesign'), 'Network (constructor): design must be of NetworkDesign class');
            assert(isa(input, 'NetworkInput'), 'Network (constructor): design must be of NetworkDesign class');
            assert(length(input)==length(design),'Network (constructor): input and design must be same length'); 
            
            %Create a single network for each input/design pair 
            if length(input)==1 && length(design)==1
                obj.network_design=design;
                obj.network_input=input;
                obj.network_output = NetworkOutput;
                obj.fwd_prop; %Updates obj(i).output
                return;
                
            %We are passed a vector of inputs/designs: recursively call
            %for each pair. 
            else
                n=1; 
                for i=1:length(design)
                    obj(n) = Network(input(i), design(i));
                    n=n+1; 
                end
            end
        end
        
        %Make one gradient descent step in this network
        function stepLearnWeights(obj, num_iterations)
        end
        
        %Compute current outputs at all layers
        function fwd_prop(obj)
            return;
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

