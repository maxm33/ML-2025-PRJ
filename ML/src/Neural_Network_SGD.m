function score = Neural_Network_SGD(numHidden1, numHidden2, eta, lambda, alpha)
    %% ===================================
    % LOADING TRAINING DATA (500 patterns)
    % ====================================
    Dataset_TR = readtable('../data/TR/ML-CUP25-TR.csv');
    
    inputs_TR  = Dataset_TR{:, 2:13};
    outputs_TR = Dataset_TR{:, 14:end};
    
    N = size(inputs_TR, 2);                     % 12 inputs
    M = size(outputs_TR, 2);                    % 4 outputs
    
    % Hyper Parameters
    % numHidden1                                % # of units inside Hidden Layer 1
    % numHidden2                                % # of units inside Hidden Layer 2
    % eta                                       % initial Learning Rate
    % lambda                                    % factor for L1 Regularization
    % alpha                                     % factor for Heavy-Ball Momentum

    eta_current = eta;

    % Early Stopping
    patience = 200;              
    tolerance = 0.01;   
    maxEpochs = 10000;
    
    %% ===================================
    % K-FOLD CROSS-VALIDATION
    % ====================================
    k = 5;
    cv = cvpartition(size(inputs_TR,1), 'KFold', k);
    
    input_layer_initial  = cell(1,k);
    hidden_layer1_initial = cell(1,k);
    hidden_layer2_initial = cell(1,k);
    output_layer_initial = cell(1,k);

    rmse_train = nan(maxEpochs, k);
    rmse_val   = nan(maxEpochs, k);
    mee_val    = nan(maxEpochs, k);
    mee_val_norm = nan(maxEpochs, k);
    mee_val_smooth = nan(maxEpochs, k);
    mee_val_norm_smooth = nan(maxEpochs, k);
    best_mee_val_smooth = nan(1, k);
    best_epoch = nan(1, k);

    training_start_time = posixtime(datetime('now'));

    for fold = 1:k
        train_idx = training(cv,fold);
        validation_idx = test(cv,fold);
        
        A_train_raw = inputs_TR(train_idx,:);
        A_val_raw   = inputs_TR(validation_idx,:);
        
        B_train = outputs_TR(train_idx,:);
        B_validation = outputs_TR(validation_idx,:);
        
        P_train = size(A_train_raw,1);
        P_val   = size(A_val_raw,1);
        
        %% ===================================
        % I/O NORMALIZATION (zero-mean / unit-variance)
        % ====================================
        mu_A  = mean(A_train_raw,1);
        std_A = std(A_train_raw,0,1);
        std_A = max(std_A,1e-8);
        
        A_train = (A_train_raw - mu_A) ./ std_A;
        A_validation = (A_val_raw - mu_A) ./ std_A;
        
        mu_B  = mean(B_train,1);
        std_B = std(B_train,0,1);
        std_B = max(std_B,1e-8);
        
        B_train_norm = (B_train - mu_B) ./ std_B;
        B_val_norm   = (B_validation - mu_B) ./ std_B;

        best_val_mee_norm = inf;
        epochs_since_improvement = 0;
        
        %% ===================================
        % NEURAL NETWORK CONFIGURATION (fully connected)
        % ====================================
        for i = 1:N
            input_layer(i) = neuron_input_unit(0);
        end
        
        for i = 1:numHidden1
            hidden_layer1(i) = neuron_hidden_unit(generate_hidden_conns_from(input_layer));
        end

        for i = 1:numHidden2
            hidden_layer2(i) = neuron_hidden_unit(generate_hidden_conns_from(hidden_layer1));
        end
        
        for i = 1:M
            output_layer(i) = neuron_output_unit(generate_output_conns_from(hidden_layer2));
        end
        
        % Saving initial weights configuration
        input_layer_initial{fold}  = input_layer;
        hidden_layer1_initial{fold} = hidden_layer1;
        hidden_layer2_initial{fold} = hidden_layer2;
        output_layer_initial{fold} = output_layer;

        %% ===================================
        % BACKPROPAGATION TRAINING LOOP
        % ====================================

        % Initialize velocity for weights and biases
        v_hidden1_bias  = zeros(1, numHidden1);
        v_hidden1_weights = cell(1, numHidden1);
        for j = 1:numHidden1
            v_hidden1_weights{j} = zeros(1,N);          % one per input -> hidden1 connection
        end

        v_hidden2_bias  = zeros(1, numHidden2);
        v_hidden2_weights = cell(1, numHidden2);
        for j = 1:numHidden2
            v_hidden2_weights{j} = zeros(1,numHidden1); % one per hidden1 -> hidden2 connection
        end
        
        v_output_bias = zeros(1,M);
        v_output_weights = cell(1,M);
        for m = 1:M
            v_output_weights{m} = zeros(1,numHidden2);  % one per hidden2 -> output connection
        end

        epoch = 0;
        while epoch < maxEpochs
            epoch = epoch + 1;

            % Shuffling patterns
            perm = randperm(P_train);
            A_train = A_train(perm,:);
            B_train_norm = B_train_norm(perm,:);
        
            % Loop over all patterns
            Ytr = zeros(P_train,M);
            for p = 1:P_train
                %% Feedforward phase
                for i = 1:N
                    input_layer(i).output = A_train(p,i); % load pattern p
                end
                for j = 1:numHidden1
                    hidden_layer1(j).compute();
                end
                for j = 1:numHidden2
                    hidden_layer2(j).compute();
                end
                outputs = zeros(1,M);
                for m = 1:M
                    output_layer(m).compute();
                    outputs(m) = output_layer(m).output;
                end
                Ytr(p,:) = outputs;
        
                %% BackPropagation phase

                % Output signals
                output_signals = (outputs - B_train_norm(p,:));
        
                % Hidden layer 2 signals
                hidden2_signals = zeros(1,numHidden2);
                for j = 1:numHidden2
                    summation = 0;
                    for k = 1:M
                        summation = summation + output_signals(k) * output_layer(k).input_connections(j).weight;
                    end
                    hidden2_signals(j) = summation * hidden_layer2(j).Leaky_ReLU_derivative(hidden_layer2(j).net);
                end

                % Hidden layer 1 signals
                hidden1_signals = zeros(1,numHidden1);
                for j = 1:numHidden1
                    summation = 0;
                    for k = 1:numHidden2
                        summation = summation + hidden2_signals(k) * hidden_layer2(k).input_connections(j).weight;
                    end
                    hidden1_signals(j) = summation * hidden_layer1(j).Leaky_ReLU_derivative(hidden_layer1(j).net);
                end
        
                %% Weights Update
                
                % Hidden2 -> Output
                for m = 1:M
                    % update bias velocity
                    v_output_bias(m) = alpha * v_output_bias(m) - eta_current * output_signals(m);
                    output_layer(m).bias_weight = output_layer(m).bias_weight + v_output_bias(m);
                
                    for j = 1:numHidden2
                        grad = output_signals(m) * hidden_layer2(j).output;
                        v_output_weights{m}(j) = alpha * v_output_weights{m}(j) - eta_current * grad;
                        output_layer(m).input_connections(j).weight = output_layer(m).input_connections(j).weight + v_output_weights{m}(j)...
                            - eta_current * lambda * sign(output_layer(m).input_connections(j).weight);
                    end
                end

                % Hidden1 -> Hidden2
                for j = 1:numHidden2
                    v_hidden2_bias(j) = alpha * v_hidden2_bias(j) - eta_current * hidden2_signals(j);
                    hidden_layer2(j).bias_weight = hidden_layer2(j).bias_weight + v_hidden2_bias(j);
                
                    for i = 1:numHidden1
                        grad = hidden2_signals(j) * hidden_layer1(i).output;
                        v_hidden2_weights{j}(i) = alpha * v_hidden2_weights{j}(i) - eta_current * grad;
                        hidden_layer2(j).input_connections(i).weight = hidden_layer2(j).input_connections(i).weight + v_hidden2_weights{j}(i)...
                            - eta_current * lambda * sign(hidden_layer2(j).input_connections(i).weight);
                    end
                end

                % Input -> Hidden1
                for j = 1:numHidden1
                    v_hidden1_bias(j) = alpha * v_hidden1_bias(j) - eta_current * hidden1_signals(j);
                    hidden_layer1(j).bias_weight = hidden_layer1(j).bias_weight + v_hidden1_bias(j);
                
                    for i = 1:N
                        grad = hidden1_signals(j) * input_layer(i).output;
                        v_hidden1_weights{j}(i) = alpha * v_hidden1_weights{j}(i) - eta_current * grad;
                        hidden_layer1(j).input_connections(i).weight = hidden_layer1(j).input_connections(i).weight + v_hidden1_weights{j}(i)...
                            - eta_current * lambda * sign(hidden_layer1(j).input_connections(i).weight);
                    end
                end
            end
        
            %% Validation
            Yval = zeros(P_val,M);
            for p = 1:P_val
                for i = 1:N
                    input_layer(i).output = A_validation(p,i); % load pattern p
                end
                for j = 1:numHidden1
                    hidden_layer1(j).compute();
                end
                for j = 1:numHidden2
                    hidden_layer2(j).compute();
                end
                for m = 1:M
                    output_layer(m).compute();
                    Yval(p,m) = output_layer(m).output;
                end
            end
        
            % RMSE computation
            err_tr = B_train_norm - Ytr;
            err_vl = B_val_norm   - Yval;
            rmse_per_output = sqrt(mean(err_tr.^2, 1));
            rmse_train(epoch,fold) = mean(rmse_per_output);
            rmse_val(epoch,fold)   = mean(sqrt(mean(err_vl.^2,1)));
        
            % MEE computation
            Yval_den = Yval .* std_B + mu_B;
            diff_val = B_validation - Yval_den;
            mee_val(epoch,fold) = mean(sqrt(sum(diff_val.^2,2)));
            w = 10;  % smoothing window
            mee_val_smooth(epoch,fold) = mean(... 
                mee_val(max(1,epoch-w+1):epoch,fold) );
            % Normalized MEE (for plotting)
            diff_val_norm = B_val_norm - Yval;
            mee_val_norm(epoch,fold) = mean(sqrt(sum(diff_val_norm.^2,2)));
            mee_val_norm_smooth(epoch,fold) = mean(... 
                mee_val_norm(max(1,epoch-w+1):epoch,fold));
        
            fprintf('MODEL -> h1 = %d; h2 = %d; eta = %g; lambda = %g; alpha = %g\n',...
                numHidden1, numHidden2, eta, lambda, alpha);
            fprintf('Epoch %d | Fold %d | RMSE_TR (norm) = %.6f | RMSE_VL (norm) = %.6f | MEE (og scale) = %.6f\n',...
                epoch, fold, rmse_train(epoch, fold), rmse_val(epoch, fold), mee_val(epoch, fold));
            disp(rmse_per_output);
        
            % If it's diverging, stop
            if isnan(mee_val(epoch,fold)) || isinf(mee_val(epoch,fold))
                fprintf('MODEL -> h1 = %d; h2 = %d; eta = %g; lambda = %g; alpha = %g\n',...
                    numHidden1, numHidden2, eta, lambda, alpha);
                fprintf("NaN detected at epoch %d fold %d — stopping model\n\n", epoch, fold);
                break;
            end
        
            current_val_mee_norm = mee_val_norm_smooth(epoch,fold);
            if current_val_mee_norm < best_val_mee_norm * (1 - tolerance)
                best_val_mee_norm = current_val_mee_norm;
                best_mee_val_smooth(fold) = mee_val_smooth(epoch,fold);
                best_epoch(fold) = epoch;
                epochs_since_improvement = 0;
            else
                epochs_since_improvement = epochs_since_improvement + 1;
            end
        
            if epochs_since_improvement >= patience
                break;
            end
        end
        input_layer_final{fold}  = input_layer;
        hidden_layer1_final{fold} = hidden_layer1;
        hidden_layer2_final{fold} = hidden_layer2;
        output_layer_final{fold} = output_layer;
    end

    training_end_time = posixtime(datetime('now'));

    % Saving model's data
    model.rmse_train_min = min(nanmean(rmse_train, 2));
    model.rmse_val_min   = min(nanmean(rmse_val, 2));
    model.best_mee_per_fold = best_mee_val_smooth;
    model.best_epoch_per_fold = best_epoch;
    model.mee_cv_mean = mean(best_mee_val_smooth);
    model.mee_cv_std  = std(best_mee_val_smooth);

    model.eta = eta;
    model.alpha = alpha;
    model.lambda = lambda;
    model.numHidden1 = numHidden1;
    model.numHidden2 = numHidden2;

    model.input_layer_initial  = input_layer_initial;
    model.hidden_layer1_initial = hidden_layer1_initial;
    model.hidden_layer2_initial = hidden_layer2_initial;
    model.output_layer_initial = output_layer_initial;

    model.input_layer_final  = input_layer_final;
    model.hidden_layer1_final = hidden_layer1_final;
    model.hidden_layer2_final = hidden_layer2_final;
    model.output_layer_final = output_layer_final;

    model.training_time = training_end_time - training_start_time;

    if ~exist('models', 'dir')
        mkdir('models');
    end

    filename = sprintf( ...
        'models/h1-%d-h2-%d-eta-%g-lambda-%g-alpha-%g_%d.mat', ...
        numHidden1, numHidden2, eta, lambda, alpha, randi(1e6));
    save(filename, 'model');

    % Plotting and saving the learning curve
    mean_train_curve = nanmean(rmse_train,2);
    mean_val_curve   = nanmean(rmse_val,2);
    mean_mee_smooth  = nanmean(mee_val_norm_smooth,2);
    
    max_trained_epoch = max(sum(~isnan(rmse_train),1));
    
    [~, name, ~] = fileparts(filename);
    plot_file = fullfile('models', [name '_plot.png']);
    
    fig = figure('Visible','off');
    hold on;

    plot(1:max_trained_epoch, ...
         mean_train_curve(1:max_trained_epoch), ...
         'b', 'LineWidth', 2); 
    plot(1:max_trained_epoch, ...
         mean_val_curve(1:max_trained_epoch), ...
         'r', 'LineWidth', 2);
    plot(1:max_trained_epoch, ...
         mean_mee_smooth(1:max_trained_epoch), ...
         'g', 'LineWidth', 2);
    
    xlabel('Epoch'); ylabel('Error');
    title(sprintf(['Learning Curves | h1 = %d; h2 = %d; eta = %g; ', ...
                   'lambda = %g; alpha = %g; SGD'], ...
            numHidden1, numHidden2, eta, lambda, alpha));
    legend({'RMSE TR (norm)', ...
            'RMSE VL (norm)', ...
            'MEE VL (norm - smoothed)'}, ...
            'Location', 'best');
    grid on;
    
    exportgraphics(fig, plot_file);
    close(fig);

    score = mean(best_mee_val_smooth);
%% 
    function hidden_conns = generate_hidden_conns_from(input_units)
        % He-Kaiming Initialization
        fan_in = numel(input_units);
        kaiming = sqrt(2 / fan_in);
    
        hidden_conns(1, fan_in) = struct('neuron',[],'weight',[]);
        for n = 1:fan_in
            hidden_conns(n).neuron = input_units(n);
            hidden_conns(n).weight = randn * kaiming;
        end
    end
    
    function output_conns = generate_output_conns_from(hidden_units)
        % Xavier Initialization
        fan_in = numel(hidden_units);
        xavier = sqrt(1 / fan_in);
    
        output_conns(1, fan_in) = struct('neuron',[],'weight',[]);
        for n = 1:fan_in
            output_conns(n).neuron = hidden_units(n);
            output_conns(n).weight = randn * xavier;
        end
    end
end
