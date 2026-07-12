function Plot(rmse_train, rmse_val, rmse_test, avg_best_metric, plot_file)
    % Calculating mean curves for train, validation, and test RMSE
    epochs_per_fold = sum(~isnan(rmse_train),1);
    max_common_epoch = min(epochs_per_fold);
    mean_train_curve = mean(rmse_train(1:max_common_epoch,:), 2, 'omitnan');
    mean_val_curve   = mean(rmse_val(1:max_common_epoch,:),   2, 'omitnan');
    mean_test_curve  = mean(rmse_test(1:max_common_epoch,:), 2, 'omitnan');
        
    % Plotting
    fig = figure('Visible','off'); hold on;
    h1 = plot(1:max_common_epoch, mean_train_curve, 'b', 'LineWidth', 1);
    h2 = plot(1:max_common_epoch, mean_val_curve, 'r', 'LineWidth', 1);
    h3 = plot(1:max_common_epoch, mean_test_curve, 'g', 'LineWidth', 1); 
    yl = ylim;
    ylim([yl(1), 1]);
    yticks(yl(1):0.05:1);

    xlabel('Epoch'); ylabel('RMSE');
    title(sprintf('Learning Curve | AVG BEST RMSE=%.3f', avg_best_metric));
    grid on;
    legend([h1 h2 h3], {'Training', 'Validation', 'Test'}, 'Location', 'northeast', 'FontSize',10);
        
    exportgraphics(fig, plot_file);
    close(fig);
end