titles={'Predictive performance on test data: minFunc';...
    'Predictive performance on test data: gradient descent';...
    'Predictive performance on test data: closed form'};


for i=1:length(alltheta)
subplot (3,1,i);

% Print out test RMS error
actual_prices = test.y;
predicted_prices = theta'*test.X;
test_rms=sqrt(mean((predicted_prices - actual_prices).^2));
fprintf('RMS testing error: %f\n', test_rms);

% Plot predictions on test data.
[actual_prices, I] = sort(actual_prices);
predicted_prices=predicted_prices(I);
plot(actual_prices, 'rx');

hold on;
plot(predicted_prices,'bx');
title(titles{i});
legend('Actual Price', 'Predicted Price');
xlabel('House #', 'interpreter','tex');
ylabel('House price ($1000s)', 'interpreter','tex');

end

saveas(gcf,'Predictive Performance','pdf');