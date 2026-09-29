function PlotTrainingLoss(loss_history, plot_file)
    fig = figure('Visible', 'off');

    semilogy(loss_history, 'LineWidth', 1.2);
    ylabel('Training Loss (log scale)');
    ylim([0 1]);

    xlabel('Epoch');

    title('Training Loss over epochs');

    grid on;
    
    exportgraphics(fig, plot_file);
    close(fig);
end