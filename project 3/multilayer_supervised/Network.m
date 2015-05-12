classdef Network < matlab.mixin.Copyable
    %NETWORK
    
    properties
        network_design
        network_input
        network_output
    end
    
    properties(Constant)
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
                fprintf('Network.train: training Network %d/%d\n', i, length(obj));
                while obj(i).checkConvergence || ~(n > obj.MAXITER)
                    obj(i).backProp(obj(i).network_input.folds);
                    n=n+1;
                end
            end
        end
        
        %Test the network on test_input
        function test(obj)
        end
        
        %Function handle to this network's cost (for minfunc)
        function cost_func = costFunc(obj)
            cost_func=@obj.cost;
        end
        
        %Make Network objects serializable
        function saveObj(obj)
        end
        
    end
    %%
    methods(Hidden=true)
        function backProp(obj)
            for i=1:length(obj.network_input.folds)
            end
        end
        
        %2.b - Check gradient using numerical approximations (unit test for gradient)
        function checkGradient(obj)
            return;
        end
        
        %Check if the loss function is changing
        function converged = checkConvergence(obj)
            converged=0;
            dif=abs(obj.network_output.loss(end)- obj.network_output.loss(end-1));
            if dif < 1e-7
                converged=1;
            end
        end
        
        %Make Network objects serializable
        function loadObj
        end
        
        function gradient(obj)
            % Initialize gradient for kth output node
            grad = zeros(1, length());
        end
    end
    
end

