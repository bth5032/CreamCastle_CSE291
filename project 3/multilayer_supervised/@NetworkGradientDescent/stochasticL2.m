function stochasticL2( obj, network)
%STOCHASTIC stochastic gradient descent with L2 regularization 
lambda=options.lambda

folds=network.network_input.folds; 
%Do grid search over lambda for all folds

%Average over folds, then pick lambda with lowest cost 
lambda = folds.getOptimalLambda;

end

