classdef NetworkOutput < matlab.mixin.Copyable
    %NETWORKOUTPUT 
    
    properties
        network %Back link
        
        stack
        act_stack
        delta_stack
        
        loss

        steps=0; 
    end
    
    methods
        
        %Nothing special for constructor
        function obj = NetworkOutput(network)
            obj.stack = initialize_weights(network.network_design.ei);
            obj.stack = stack2params(obj.stack);
            obj.network=network;
        end
            
        %Plot the convergence of the Network 
        function h = plot_convergence(obj)
        end
        
        %Plot log(pdf(Data))
        function h = plot_log_liklihood(obj)
        end
        
        function h = plot_loss(obj)
        end     
    end     
end

