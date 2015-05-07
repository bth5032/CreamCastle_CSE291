classdef NetworkOutput
    %NETWORKOUTPUT 
    
    properties
        network %Back link
        
        gradient
        log_liklihood
        loss
        weights
    end
    
    methods
        %Plot the convergence of the Network 
        function h = plot_convergence(obj)
            obj.network.
        end
        
        %Plot log(pdf(Data))
        function h = plot_log_liklihood(obj)
        end

        
        function h = plot_loss(obj)
        end
        
    end     
end

