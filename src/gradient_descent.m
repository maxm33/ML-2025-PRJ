function W = gradient_descent(A, B, eta, lambda, epochs)

    N = size(A, 1);
    MSE_history = zeros(500,1);

    W = (rand(size(A,2), size(B,2)) - 0.5) * 0.02;  % weights initialization
    
    for epoch = 1:epochs
        MSE = 0;
    
        for p = 1:N
            xp = A(p,:);
            yp = B(p,:);
    
            hp = xp * W;                            % feeding pattern p
    
            % Mean Square Error
            err = yp - hp;
            MSE = MSE + mean(err.^2);
    
            delta_w = -2 * xp' * err;               % gradient computation
    
            W = W - eta * delta_w - lambda * W;     % weights update rule
        end
    
        % Average MSE for the epoch
        MSE_history(epoch) = MSE / N;
    end
    
    figure;
    plot(MSE_history, 'LineWidth', 2);
    xlabel('Epoch'); ylabel('Mean Square Error');
    title('Training MSE over Epochs');
    axis([0 epochs 0.5 1])
    grid on;

end
