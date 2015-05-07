classdef Network
    %NETWORK
    
    properties
        network_input
        network_output
        network_weights
        network_converged
        
    end
    
    methods
        %Constructor
        function obj = Network()
        end
        
        %Make one gradient descent step in this network
        function stepLearnWeights(obj, num_iterations)
        end       
        
        %Compute gradient of network
        function gradient(obj)
        end
        
        function makeNetwork(obj)
        end
        
        function loss(obj)
        end
        
        %2.b - Check gradient using numerical approximations (unit test for gradient)
        function checkGradient(obj)
        end
        
    end
end

