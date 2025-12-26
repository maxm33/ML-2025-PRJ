%% ===================================
% LOADING TRAINING DATA (500 patterns)
% ====================================
Dataset_TR = readtable('../data/TR/ML-CUP25-TR.csv');

inputs_TR  = Dataset_TR{:, 2:13};           % 12 inputs
outputs_TR = Dataset_TR{:, 14:end};         % 4 outputs

%% ===================================
% I/O NORMALIZATION (zero-mean / unit-variance)
% ====================================

% Inputs
mu_in  = mean(inputs_TR, 1);
std_in = std(inputs_TR, 0, 1);

A = (inputs_TR - mu_in) ./ std_in;

A = [ones(size(A,1),1) A];                  % added bias column

% Outputs
mu_out  = mean(outputs_TR, 1);
std_out = std(outputs_TR, 0, 1);

B = (outputs_TR - mu_out) ./ std_out;

%% ===================================
% WEIGHTS CALCULATION
% ====================================

% Hyper-Parameters
eta = 1e-6;                                 % learning rate factor
lambda = 1e-8;                              % L1 regularization factor
epochs = 300;

W_online = online_gradient_descent(A,B,1e-6,lambda,epochs);

W_batch = batch_gradient_descent(A,B,1e-3,lambda,epochs);

%% Online
function W = online_gradient_descent(inputs, outputs, eta, lambda, epochs)

    N = size(inputs, 1);
    MSE_history = zeros(epochs,1);

    % Randomized Weights Initialization
    W = (rand(size(inputs,2), size(outputs,2)) - 0.5) * 0.02;

    disp("Random Initial Weights:");
    disp(W);
    
    for epoch = 1:epochs
        MSE = 0;
    
        for p = 1:N
            xp = inputs(p,:);
            yp = outputs(p,:);
    
            hp = xp * W;                                  % feeding pattern p
    
            % Mean Square Error
            err = yp - hp;
            MSE = MSE + mean(err.^2);
    
            delta_w = - 2 * xp' * err;                    % gradient computation
    
            W = W - eta * delta_w - lambda * sign(W);     % weights update rule
        end
    
        % Average MSE for the epoch
        MSE_history(epoch) = MSE / N;
    end
    
    figure;
    plot(MSE_history, 'LineWidth', 2);
    xlabel('Epoch'); ylabel('Mean Square Error');
    title('Training MSE over Epochs (Online GD)');
    axis([0 epochs 0 1])
    grid on;
end

%% Batch
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
    
            hp = xp * W;                                      % feeding pattern p
    
            % Mean Square Error
            err = yp - hp;
            MSE = MSE + mean(err.^2);
    
            delta_w = delta_w - 2/N * xp' * err;              % gradient computation
    
        end

        W = W - eta * delta_w - lambda * sign(W);             % weights update rule
    
        % Average MSE for the epoch
        MSE_history(epoch) = MSE / N;
    end

    figure;
    plot(MSE_history, 'LineWidth', 2);
    xlabel('Epoch'); ylabel('Mean Square Error');
    title('Training MSE over Epochs (Batch GD)');
    axis([0 epochs 0 1])
    grid on;
end
