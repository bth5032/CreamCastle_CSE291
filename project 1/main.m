addpath(fullfile(pwd,'stanford_dl_ex-master','ex1'));
addpath(fullfile(pwd,'stanford_dl_ex-master','common','minFunc_2012','minFunc'));
addpath(fullfile(pwd,'stanford_dl_ex-master','common','minFunc_2012','minFunc','compiled'));

ex1a_linreg;

titles={'Predictive performance on test data: minFunc';...
    'Predictive performance on test data: gradient descent';...
    'Predictive performance on test data: closed form'};

method={ 'minFunc', 'gradient descent','closed form'};
    
for i=1:length(alltheta)
    subplot (3,1,i);
    
    %Print out test RMS error
    actual_prices = test.y;
    predicted_prices = alltheta{i}'*test.X;
    test_rms=sqrt(mean((predicted_prices - actual_prices).^2));
    disp([method{i} ' RMS testing error: %f']);
    xlabel('House #');
    ylabel('House price ($1000s)');
    
    %Plot predictions on test data.
    [actual_prices, I] = sort(actual_prices);
    predicted_prices=predicted_prices(I);
    plot(actual_prices, 'rx');
    xlabel('House #');
    ylabel('House price ($1000s)');
    
    hold on;
    plot(predicted_prices,'bx');
    title(titles{i});
    legend('Actual Price', 'Predicted Price');
    xlabel('House #');
    ylabel('House price ($1000s)');
end

saveas(gcf,'Predictive Performance','pdf');
