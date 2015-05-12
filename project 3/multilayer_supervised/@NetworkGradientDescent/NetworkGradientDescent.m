classdef NetworkGradientDescent < matlab.mixin.Copyable
    %NETWORKGRADIENTDESCENT Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        obj_funcs
        options
    end
    
    properties(Constant)
        MAXITER=1000;
    end
    
    methods
        %Constructor
        function obj=NetworkGradientDescent
            %Options for objective functions
            obj.obj_funcs.BATCH_L2=1;
            
            %Options for optimization
            obj.options.display = 'iter';
            obj.options.maxFunEvals = 1e4;
            obj.options.Method = 'lbfgs';
        end
        
        %Return function handle to desired objective function 
        function objective_func = objective(obj, obj_id)
            if obj.obj_funcs.BATCH_L2==obj_id
                objective_func=@NetworkGradientDescent.batchL2;
            end
        end
        
        %Check to make sure objective functions have uniform interfaces
        function verifyObjectiveInterfaces(obj)
            return
        end
    end
    
    methods(Hidden=true)
        function folds = crossValidate(obj, objective_func, lambda, numfolds)
            folds=[];
        end
    end
    
    
    methods(Static=true)
        function interface = getObjectiveInterface
            interface.input = {'theta', 'ei', 'data', 'labels', 'pred_only'};
            interface.output = {'cost', 'grad', 'pred_prob'};
        end
        
        
        %Declare all objective functions (in files)
        function batchL2 end
        function batchGaussNewtonL2 end
        function stochasticL2 end
        function stochasticGaussNewtonL2 end
    end
    
end

