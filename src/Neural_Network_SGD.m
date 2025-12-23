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
    numHidden = 50;                             % # of units inside Hidden Layer
    
    eta = 1e-2;                                 % Learning Rate
    
    lambda = 1e-5;                              % factor for L1 Regularization
    
    epochs = 1000;
    
    %% ===================================
    % K-FOLD 
    % ====================================
    k = 5;
    cv = cvpartition(size(inputs_TR,1), 'KFold', k);

    for fold = 1:k
        train_idx = training(cv,fold); 
        validation_idx = test(cv,fold);      
    
        A_train = inputs_TR(train_idx, :);
        A_validation = inputs_TR(validation_idx, :);    
    
        B_train = outputs_TR(train_idx, :);
        B_validation = outputs_TR(validation_idx, :);
    
        P_train = size(A_train,1);
        P_val = size(A_validation,1);

        %% ===================================
        % I/O NORMALIZATION (zero-mean / unit-variance)
        % ====================================
        A_train_raw = inputs_TR(train_idx,:);
        A_val_raw   = inputs_TR(validation_idx,:);

        mu_A  = mean(A_train_raw,1);
        std_A = std(A_train_raw,0,1);

        A_train = (A_train_raw - mu_A) ./ std_A;
        A_validation   = (A_val_raw   - mu_A) ./ std_A;

        mu_B  = mean(B_train,1);
        std_B = std(B_train,0,1);

        B_train_norm = (B_train - mu_B) ./ std_B;
        B_val_norm   = (B_validation - mu_B) ./ std_B;

        %% ===================================
        % NEURAL NETWORK CONFIGURATION (fully connected)
        % ====================================
        
        % Input Layer
        for i = 1:N
            input_layer(i) = neuron_input_unit(0);
        end
        
        % Hidden Layer
        for i = 1:numHidden
            hidden_layer(i) = neuron_hidden_unit(generate_hidden_conns_from(input_layer));
        end
        
        % Output Layer
        for i = 1:M
            output_layer(i) = neuron_output_unit(generate_output_conns_from(hidden_layer));
        end
        
        %% ===================================
        % BACKPROPAGATION TRAINING LOOP
        % ====================================
        
        epoch_times = zeros(1, epochs);
        rmse_history = zeros(1, epochs);
        
        % Plot Initialization
        figure;
        hTrain = plot(NaN, NaN, 'b-', 'LineWidth', 2); hold on;
        hVal   = plot(NaN, NaN, 'r--', 'LineWidth', 2); 
        xlabel('Epoch'); ylabel('RMSE'); 
        title('Learning Curve'); grid on;
        legend('Train RMSE', 'Validation RMSE');
    
        for epoch = 1:epochs
            epoch_time_start = posixtime(datetime('now'));
    
            total_error = 0;
            Yhat = zeros(P_train, M);
    
            % Loop over all patterns 
            for p = 1:P_train
                %% Feedforward phase
                for i = 1:N
                    input_layer(i).output = A_train(p,i); % load pattern p
                end
                for i = 1:numHidden
                    hidden_layer(i).compute();
                end
                outputs = zeros(1,M);
                for i = 1:M
                    output_layer(i).compute();
                    outputs(i) = output_layer(i).output;
                    Yhat(p, :) = outputs;
                end

                outputs_denorm = outputs .* std_B + mu_B;
                denorm_diff = B_train(p,:) - outputs_denorm;
                total_error = total_error + sqrt(sum(denorm_diff.^2));

                %% Output signals
                output_signals = zeros(1, M);
                for k = 1:M
                    output_signals(k) = (B_train_norm(p,k) - outputs(k));
                end
    
                %% Hidden signals
                hidden_signals = zeros(1, numHidden);
                for j = 1:numHidden
                    summation = 0;
            
                    for k = 1:M
                        summation = summation + ...
                            output_signals(k) * ...
                            output_layer(k).input_connections(j).weight;
                    end
            
                    hidden_signals(j) = ...
                        summation * hidden_layer(j).Leaky_ReLU_derivative(hidden_layer(j).net);
                end
    
                %% Update kj weights (Hidden -> Output)
                for k = 1:M
                    output_layer(k).bias_weight = ...
                        output_layer(k).bias_weight + eta * output_signals(k);
            
                    for j = 1:numHidden
                        output_layer(k).input_connections(j).weight = ...
                            output_layer(k).input_connections(j).weight + ...
                            eta * output_signals(k) * hidden_layer(j).output - ...
                            lambda * sign(output_layer(k).input_connections(j).weight);
                    end
                end
    
                %% Update ji weights (Input -> Hidden)
                for j = 1:numHidden
                    hidden_layer(j).bias_weight = ...
                        hidden_layer(j).bias_weight + eta * hidden_signals(j);
            
                    for i = 1:numel(hidden_layer(j).input_connections)
                        hidden_layer(j).input_connections(i).weight = ...
                            hidden_layer(j).input_connections(i).weight + ...
                            eta * hidden_signals(j) * ...
                            hidden_layer(j).input_connections(i).neuron.output - ...
                            lambda * sign(hidden_layer(j).input_connections(i).weight);
                    end
                end
            end

            Yval = zeros(P_val, M);
            for p = 1:P_val
                % Load input pattern
                for i = 1:N
                    input_layer(i).output = A_validation(p,i);
                end
                
                % Forward pass Hidden Layer
                for i = 1:numHidden
                    hidden_layer(i).compute();
                end
                
                % Forward pass Output Layer
                outputs = zeros(1,M);
                for i = 1:M
                    output_layer(i).compute();
                    outputs(i) = output_layer(i).output;
                    Yval(p,:) = outputs;
                end
            end
            err_val = B_val_norm - Yval;
            rmse_val_per_output = sqrt(mean(err_val.^2,1));
            rmse_val(epoch) = mean(rmse_val_per_output);

            % RMSE with collected network outputs over an epoch
            err = B_train_norm - Yhat;
            rmse_per_output = sqrt(mean(err.^2, 1));
        
            % Compute Mean Euclidian Error over an epoch
            mee = total_error / P_train;
            rmse_history(epoch) = mean(rmse_per_output);
            fprintf('Epoch %d | RMSE (norm) = %.6f | MEE (og scale) = %.6f\n', epoch, rmse_history(epoch), mee);
            disp(rmse_per_output);
        
            % Live Plot
            set(hTrain,'XData',1:epoch,'YData',rmse_history(1:epoch));
            set(hVal,   'XData', 1:epoch, 'YData', rmse_val(1:epoch));
            drawnow;
        
            % Shuffling the patterns
            %perm = randperm(size(A,1));         % random rows order
            %A = A(perm, :);
            %B = B(perm, :);
        
            epoch_times(epoch) = posixtime(datetime('now')) - epoch_time_start;
        end
    end

    fprintf('Total Training Time (seconds) = %.3f | Average Epoch Time (seconds) = %.3f\n', sum(epoch_times), mean(epoch_times));
    %%
    function hidden_conns = generate_hidden_conns_from(input_units)
        hidden_conns(1, numel(input_units)) = struct('neuron',[],'weight',[]);
    
        for i = 1:numel(input_units)
            hidden_conns(i).neuron = input_units(i);
            hidden_conns(i).weight = randn * 0.1;
        end
    end
    
    function output_conns = generate_output_conns_from(hidden_units)
        output_conns(1, numel(hidden_units)) = struct('neuron',[],'weight',[]);
    
        for i = 1:numel(hidden_units)
            output_conns(i).neuron = hidden_units(i);
            output_conns(i).weight = randn * 0.1;
        end
    end
