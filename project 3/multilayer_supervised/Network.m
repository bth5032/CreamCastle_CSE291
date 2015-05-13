classdef Network < matlab.mixin.Copyable
    %NETWORK
    
    properties
        network_design
        network_input
        network_output
    end
    
    properties(Constant)
        MAX_TRAIN_PASSES=100; 
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
            
            %obj.forwardProp;
        end
        
        %Train the network on train_input
        function train(obj)
            for i=1:length(obj)
                %Iterate until MAXITER or converged
                n=1;
                fprintf('Network.train: training Network %d/%d\n', i, length(obj));
                while obj(i).checkConvergence || ~(n > obj.MAX_TRAIN_PASSES)
                    theta=stack2params(obj(i).network_output.stack);
                    data=obj(i).network_input.features; 
                    labels=obj(i).network_input.labels; 
                    ei=obj(i).network_design.ei; 
                    
                    %Gradient Descent 
                    gd=NetworkGradientDescent; 
                    gd.stochasticL2(obj); 
                    
                    [cost, ~, output, hAct, delta] = supervised_dnn_cost( theta, ei, data, labels);
                    obj(i).network_output.output{n}=output; 
                    obj(i).network_output.act_stack{n}=param2stack(hAct); 
                    obj(i).network_output.delta_stack{n}=delta; 
                    obj(i).network_output.loss(n)=cost;
                    obj(i).network_output.steps=n;
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
        
        %Compute all network activations
        function forwardProp(obj)
            for k=1:length(obj)
                act=obj(k).network_input.features;
                afunc=obj(k).network_design.activationFunction;
                
                obj(k).network_output.stack
                for i=1:length(obj(k).network_output.stack)
                    this_weight = obj(k).network_output.stack{i};
                    act = afunc(this_weight.W*act + this_weight.b);
                    obj(k).network_output.activations{i}=act;
                end
            end
        end
        
        %2.b - Check gradient using numerical approximations (unit test for gradient)
        function checkGradient(obj)
            derivativeCheck
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
    end
end

