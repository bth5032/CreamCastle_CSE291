classdef NetworkOutput < matlab.mixin.Copyable
    %NETWORKOUTPUT 
    
    properties
        network %Back link
        
        activations
        gradient
        log_liklihood
        loss
        weights
        
        steps=0; 
    end
    
    methods
        
        %Nothing special for constructor
        function obj = NetworkOutput
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

