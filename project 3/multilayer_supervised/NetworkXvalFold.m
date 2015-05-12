classdef NetworkXvalFold
    %NETWORKXVALFOLD Breakdown of data into train and test inputs
    
    properties
        train_input
        test_input
            
        lambda
        stack
        cost
    end
    
    methods
        %Constructor
        function obj = NetworkXvalFold(train_input, test_input)
            obj.train_input=train_input;
            obj.test_input=test_input;
        end  
        
        %Get optimal result
        function [lambda, cost, stack] = getOptimalResults(obj)
            [cost, idx] = min(mean(obj.cost));
            lambda = obj.lambda(idx(1));
            stack = obj.stack(idx(1));
        end  
    end  
end