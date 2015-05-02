classdef Network
    %NETWORK
    
    properties
        network_input
        network_output
        network_weights
        network_converged
        
        %Structure containg nodes and connections
        net_tree
    end
    
    methods
        %Constructor
        function obj = NetworkOutput()
        end
        
        %Make one gradient descent step in this network
        function step_learn_weights(obj, num_iterations)
        end       
        
        %Compute gradient of network
        function gradient(obj)
        end
        
        %Recursively construct the desired network and store in obj.net_tree
        function make_network(obj)
        end
    end
    
end

