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
            
            for i=1:length(design)
                for j=1:length(input)
                    obj(i,j).network_design=design(i);
                    obj(i,j).network_input=input(i);
                    obj(i,j).network_output = NetworkOutput;
                    obj(i,j).fwd_prop; %Updates obj(i).output
                end
            end
        end
        
        %Make one gradient descent step in this network
        function stepLearnWeights(obj, num_iterations)
        end
        
        %Make one gradient descent step in this network
        function fwd_prop(obj)
            %Compute current outputs
        end
        
        %Perform backprop on network
        function gradient_desc(obj)
        end
        
        %2.b - Check gradient using numerical approximations (unit test for gradient)
        function checkGradient(obj)
        end
        
        function loss(obj)
        end
        
    end
end

