function [ theta, error ] = stochastic_grad_desc(func, theta0, alpha, maxIter, X, y, X_val, y_val, ei)
    [~, m] = size(X);
    theta = theta0;
    error = zeros(1, maxIter);
    fprintf('epoch    error\n');
    
    % Iterate until max iterations reached
    for epoch = 1:maxIter
        % Shuffle the examples on each iteration
        perm = randperm(m);
        X = X(:, perm);
        y = y(perm);
        
        % Run over the dataset, randomly permuted for each iteration
        for j = 1:m
            [~, g] = func(theta, ei, X(:, j), y(j), false);
            theta = theta - alpha * g;
        end
        
        % Predict performance with current parameter set
        [~, ~, pred] = func(theta, ei, X_val, y_val, true);
        [~,pred] = max(pred);
        error(epoch) = 1 - mean(pred == y_val);
        fprintf('% 5d    %.3f \n', epoch, error(epoch));
    end
end

