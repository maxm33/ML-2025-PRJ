function Plot_train_loss(loss_history, plot_file)
    fig = figure('Visible', 'off');
    semilogy(loss_history, 'LineWidth', 1.2);
    xlabel('Epoch');
    ylabel('Training Loss (log scale)');
    title('Training Loss Curve');
    grid on;
    exportgraphics(fig, plot_file);
    close(fig);
end