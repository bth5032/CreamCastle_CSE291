classdef NetworkXvalFold
    %NETWORKXVALFOLD Breakdown of data into train and test inputs
    
    properties
        train_input
        test_input
        
        weights
        lambda
        error
    end
    
    methods
        %Constructor
        function obj = NetworkXvalFold(train_input, test_input)
            obj.train_input=train_input;
            obj.test_input=test_input;
        end
    end  
end