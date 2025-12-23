function score = Neural_Network_batch(numHidden1, numHidden2, eta, lambda)
    %% ===================================
    % LOADING TRAINING DATA (500 patterns)
    % ====================================
    Dataset_TR = readtable('../data/TR/ML-CUP25-TR.csv');
    
    inputs_TR  = Dataset_TR{:, 2:13};
    outputs_TR = Dataset_TR{:, 14:end};
    
    N = size(inputs_TR, 2);                     % 12 inputs
    M = size(outputs_TR, 2);                    % 4 outputs
    P = size(inputs_TR, 1);                     % 500 patterns
    
    % Hyper Parameters
    % numHidden1                                % # of units inside first Hidden Layer
    % numHidden2                                % # of units inside second Hidden Layer
    
    % eta                                       % initial Learning Rate
    %eta_tau = eta / 100;                       % eta to use from epoch tau
    %tau = 200;                                 % epoch where eta will be fully decayed
    
    % lambda                                    % factor for L1 Regularization
    
    maxEpochs = 10000;
    
    %% ===================================
    % I/O NORMALIZATION (zero-mean / unit-variance)
    % ====================================
    A = (inputs_TR - mean(inputs_TR,1)) ./ std(inputs_TR,0,1);
    B = (outputs_TR - mean(outputs_TR,1)) ./ std(outputs_TR,0,1);
    
    %% ===================================
    % NEURAL NETWORK CONFIGURATION (fully connected)
    % ====================================
    
    % Input Layer
    for i = 1:N
        input_layer(i) = neuron_input_unit(0);
    end
    
    % Hidden Layer 1
    for i = 1:numHidden1
        hidden_layer1(i) = neuron_hidden_unit(generate_hidden_conns_from(input_layer, M));
    end
    
    % Hidden Layer 2
    for i = 1:numHidden2
        hidden_layer2(i) = neuron_hidden_unit(generate_hidden_conns_from(hidden_layer1, M));
    end
    
    % Output Layer
    for i = 1:M
        output_layer(i) = neuron_output_unit(generate_output_conns_from(hidden_layer2, M));
    end

    % Saving initial weights configuration
    input_layer_initial = input_layer;
    hidden_layer1_initial = hidden_layer1;
    hidden_layer2_initial = hidden_layer2;
    output_layer_initial = output_layer;
    
    %% ===================================
    % BACKPROPAGATION TRAINING LOOP
    % ====================================

    mee_history = zeros(1, maxEpochs);
    rmse_history = zeros(1, maxEpochs);
    epoch_times = zeros(1, maxEpochs);

    % Early Stopping
    patience = 200;              
    tolerance = 0.05;                           % minimum of 5% in patience epochs            
    
    % Plot Initialization
    %figure;
    %hLine = plot(NaN, NaN, 'b-', 'LineWidth', 2);
    %xlabel('Epoch'); ylabel('RMSE'); title('Learning Curve'); grid on; hold on;
    
    epoch = 0;
    while epoch < maxEpochs
        epoch = epoch + 1;
        epoch_time_start = posixtime(datetime('now'));
    
        total_error = 0;
        Yhat = zeros(P, M);
    
        % Gradient Accumulators
        grad_W_h1 = zeros(numHidden1, N); grad_b_h1 = zeros(1,numHidden1);
        grad_W_h2 = zeros(numHidden2, numHidden1); grad_b_h2 = zeros(1,numHidden2);
        grad_W_out = zeros(M,numHidden2); grad_b_out = zeros(1,M);
    
        % Loop over all patterns
        for p = 1:P
            %% Feedforward phase
            for i = 1:N
                input_layer(i).output = A(p,i); % load pattern p
            end
            for i = 1:numHidden1
                hidden_layer1(i).compute();
            end
            for i = 1:numHidden2
                hidden_layer2(i).compute();
            end
            outputs = zeros(1,M);
            for i = 1:M
                output_layer(i).compute();
                outputs(i) = output_layer(i).output;
                Yhat(p,:) = outputs;
            end
    
            denorm_diff = (B(p,:) - outputs) .* std(outputs_TR,0,1);
            total_error = total_error + sqrt(sum(denorm_diff.^2));
    
            %% BackPropagation phase
    
            % Output signals
            output_signals = zeros(1,M);
            for k = 1:M
                output_signals(k) = (B(p,k) - outputs(k));
            end
    
            % Hidden layer 2 signals
            hidden2_signals = zeros(1, numHidden2);
            for j = 1:numHidden2
                summation = 0;
                for k = 1:M
                    summation = summation + output_signals(k) * output_layer(k).input_connections(j).weight;
                end
                hidden2_signals(j) = summation * hidden_layer2(j).Leaky_ReLU_derivative(hidden_layer2(j).net);
            end
    
            % Hidden layer 1 signals
            hidden1_signals = zeros(1, numHidden1);
            for j = 1:numHidden1
                summation = 0;
                for u = 1:numHidden2
                    summation = summation + hidden2_signals(u) * hidden_layer2(u).input_connections(j).weight;
                end
                hidden1_signals(j) = summation * hidden_layer1(j).Leaky_ReLU_derivative(hidden_layer1(j).net);
            end
    
            %% Gradients Accumulation
    
            % Output layer
            for k = 1:M
                grad_b_out(k) = grad_b_out(k) + output_signals(k);
                for j = 1:numHidden2
                    grad_W_out(k,j) = grad_W_out(k,j) + output_signals(k) * hidden_layer2(j).output;
                end
            end
    
            % Hidden layer 2
            for j = 1:numHidden2
                grad_b_h2(j) = grad_b_h2(j) + hidden2_signals(j);
                for i = 1:numHidden1
                    grad_W_h2(j,i) = grad_W_h2(j,i) + hidden2_signals(j) * hidden_layer1(i).output;
                end
            end
    
            % Hidden layer 1
            for j = 1:numHidden1
                grad_b_h1(j) = grad_b_h1(j) + hidden1_signals(j);
                for i = 1:N
                    grad_W_h1(j,i) = grad_W_h1(j,i) + hidden1_signals(j) * input_layer(i).output;
                end
            end
        end
        %% Weights Update
    
        % Input -> Hidden1
        for j = 1:numHidden1
            hidden_layer1(j).bias_weight = hidden_layer1(j).bias_weight + eta * grad_b_h1(j) / P;
            for i = 1:N
                hidden_layer1(j).input_connections(i).weight = ...
                    hidden_layer1(j).input_connections(i).weight + eta * grad_W_h1(j,i) / P - ...
                        lambda * sign(hidden_layer1(j).input_connections(i).weight);
            end
        end
    
        % Hidden1 -> Hidden2
        for j = 1:numHidden2
            hidden_layer2(j).bias_weight = hidden_layer2(j).bias_weight + eta * grad_b_h2(j) / P;
            for i = 1:numHidden1
                hidden_layer2(j).input_connections(i).weight = ...
                    hidden_layer2(j).input_connections(i).weight + eta * grad_W_h2(j,i) / P - ...
                        lambda * sign(hidden_layer2(j).input_connections(i).weight);
            end
        end
    
        % Hidden2 -> Output
        for k = 1:M
            output_layer(k).bias_weight = output_layer(k).bias_weight + eta * grad_b_out(k) / P;
            for j = 1:numHidden2
                output_layer(k).input_connections(j).weight = ...
                    output_layer(k).input_connections(j).weight + eta * grad_W_out(k,j) / P - ...
                        lambda * sign(output_layer(k).input_connections(j).weight);
            end
        end
    
        % RMSE with collected network outputs over an epoch
        err = B - Yhat;
        rmse_per_output = sqrt(mean(err.^2, 1));
    
        % Compute Mean Euclidian Error over an epoch
        mee_history(epoch) = total_error / P;
        rmse_history(epoch) = mean(rmse_per_output);
        fprintf('Epoch %d | RMSE (norm) = %.6f | MEE (og scale) = %.6f\n', epoch, rmse_history(epoch), mee_history(epoch));
        disp(rmse_per_output);
    
        % Live Plot
        %set(hLine,'XData',1:epoch,'YData',rmse_history(1:epoch));
        %drawnow;
    
        % Variable (linear decaying) learning rate
        %if epoch <= tau
        %    gamma = epoch / tau;
        %    eta = (1 - gamma) * eta + gamma * eta_tau;
        %else
        %    eta = eta_tau;
        %end

        if epoch > patience
            past_rmse = rmse_history(epoch - patience);
            current_rmse = rmse_history(epoch);
        
            % Relative improvement over last patience epochs
            improvement = (past_rmse - current_rmse) / max(past_rmse, 1e-8);
        
            if improvement < tolerance
                fprintf("EARLY STOP at epoch %d\n", epoch);
                break;
            end
        end
        
        epoch_times(epoch) = posixtime(datetime('now')) - epoch_time_start;
    end
    
    fprintf('Total Training Time (seconds) = %.3f | Average Epoch Time (seconds) = %.3f\n', sum(epoch_times), mean(epoch_times));

    % Saving model's data
    model.mee_min = min(mee_history);
    model.mee_final = mee_history(end);
    model.rmse_min = min(rmse_history);
    model.rmse_final = rmse_history(end);

    model.eta = eta;
    model.lambda = lambda;
    model.numHidden1 = numHidden1;
    model.numHidden2 = numHidden2;

    model.mu_input = mean(inputs_TR,1);
    model.sigma_input = std(inputs_TR,0,1);
    model.mu_output = mean(outputs_TR,1);
    model.sigma_output = std(outputs_TR,0,1);

    model.input_layer_initial = input_layer_initial;
    model.hidden_layer1_initial = hidden_layer1_initial;
    model.hidden_layer2_initial = hidden_layer2_initial;
    model.output_layer_initial = output_layer_initial;

    model.input_layer_final = input_layer;
    model.hidden_layer1_final = hidden_layer1;
    model.hidden_layer2_final = hidden_layer2;
    model.output_layer_final = output_layer;

    model.training_time = sum(epoch_times);
    model.epoch_time = mean(epoch_times);

    if ~exist('models', 'dir')
        mkdir('models');
    end

    filename = sprintf( ...
        'models/h1-%d-h2-%d-eta-%g-lambda-%g_%d.mat', ...
        numHidden1, numHidden2, eta, lambda, randi(1e6));
    save(filename, 'model');

    % Plotting and saving the learning curve
    [~, name, ~] = fileparts(filename);
    plot_file = fullfile('models', [name '_plot.png']);
    
    fig = figure('Visible','off');
    plot(1:maxEpochs, rmse_history, 'LineWidth', 2);
    xlabel('Epoch'); ylabel('RMSE');
    title(sprintf('Learning Curve | h1 = %d; h2 = %d; eta = %g; lambda = %g; Batch', ...
        numHidden1, numHidden2, eta, lambda));
    grid on;
    
    exportgraphics(fig, plot_file);
    close(fig);

    score = mee_history(end);
    %%
    function hidden_conns = generate_hidden_conns_from(input_units, fan_out)
        hidden_conns(1, numel(input_units)) = struct('neuron',[],'weight',[]);

        % Normal Xavier Initialization
        fan_in = numel(input_units);
        xavier = sqrt(2 / (fan_in + fan_out));

        for n = 1:fan_in
            hidden_conns(n).neuron = input_units(n);
            hidden_conns(n).weight = randn * xavier;
        end
    end
    
    function output_conns = generate_output_conns_from(hidden_units, fan_out)
        output_conns(1, numel(hidden_units)) = struct('neuron',[],'weight',[]);

        % Normal Xavier Initialization
        fan_in = numel(hidden_units);
        xavier = sqrt(2 / (fan_in + fan_out));
    
        for n = 1:fan_in
            output_conns(n).neuron = hidden_units(n);
            output_conns(n).weight = randn * xavier;
        end
    end
end
