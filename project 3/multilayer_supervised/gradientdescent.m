function [ output_args ] = gradientdescent( dnn_cost_function, params, options, ei, data, labels )
%GRADIENTDESCENT Stochastic gradient descent function (with momentum)
[ cost, grad, pred_prob] = dnn_cost_function(theta, ei, data, labels, pred_only);


end

