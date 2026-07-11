function Plot(rmse_train, rmse_val, rmse_test, avg_best_metric, plot_file)
    % Calculating mean curves for train, validation, and test RMSE
    epochs_per_fold = sum(~isnan(rmse_train),1);
    max_common_epoch = min(epochs_per_fold);
    mean_train_curve = mean(rmse_train(1:max_common_epoch,:), 2, 'omitnan');
    mean_val_curve   = mean(rmse_val(1:max_common_epoch,:),   2, 'omitnan');
    mean_test_curve  = mean(rmse_test(1:max_common_epoch,:), 2, 'omitnan');
        
    % Plotting
    fig = figure('Visible','off');
    h1 = plot(1:max_common_epoch, mean_train_curve, 'b', 'LineWidth', 2); hold on;
    h2 = plot(1:max_common_epoch, mean_val_curve, 'r', 'LineWidth', 2);
    h3 = plot(1:max_common_epoch, mean_test_curve, 'g', 'LineWidth', 2); 
        
    xlabel('Epoch'); ylabel('RMSE');
    title(sprintf('Learning Curve | valMEE=%.3f', avg_best_metric));
    grid on;
    legend([h1 h2 h3], {'Train', 'Validation', 'Test'}, 'Location', 'best', 'FontSize',18);
        
    exportgraphics(fig, plot_file);
    close(fig);
end