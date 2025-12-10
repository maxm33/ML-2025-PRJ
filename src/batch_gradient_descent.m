function W = batch_gradient_descent(inputs, outputs, eta, lambda, epochs)

    N = size(inputs, 1);
    MSE_history = zeros(epochs,1);

    % Randomized Weights Initialization
    W = (rand(size(inputs,2), size(outputs,2)) - 0.5) * 0.02;

    disp("Random Initial Weights:");
    disp(W);
    
    for epoch = 1:epochs
        MSE = 0;
        delta_w = zeros(size(W));

        for p = 1:N
            xp = inputs(p,:);
            yp = outputs(p,:);
    
            hp = xp * W;                            % feeding pattern p
    
            % Mean Square Error
            err = yp - hp;
            MSE = MSE + mean(err(:).^2);
    
            delta_w = delta_w - 2/N * xp' * err;    % gradient computation
    
        end

        W = W - eta * (delta_w + lambda * W);       % weights update rule
    
        % Average MSE for the epoch
        MSE_history(epoch) = MSE / N;
    end
    
    figure;
    plot(MSE_history, 'LineWidth', 2);
    xlabel('Epoch'); ylabel('Mean Square Error');
    title('Training MSE over Epochs (Batch GD)');
    axis([0 epochs 0.5 1])
    grid on;

end
