function score = Neural_Network_SGD(numHidden, eta, lambda, alpha)
    %% ===================================
    % LOADING TRAINING DATA (500 patterns)
    % ====================================
    Dataset_TR = readtable('../data/TR/ML-CUP25-TR.csv');
    
    inputs_TR  = Dataset_TR{:, 2:13};
    outputs_TR = Dataset_TR{:, 14:end};
    
    N = size(inputs_TR, 2);                     % 12 inputs
    M = size(outputs_TR, 2);                    % 4 outputs
    
    % Hyper Parameters
    % numHidden                                 % # of units inside Hidden Layer
    % eta                                       % initial Learning Rate
    % lambda                                    % factor for L1 Regularization
    % alpha                                     % factor for Heavy-Ball Momentum

    eta_current = eta;
    eta_tau = eta / 10;                         % eta to use from epoch tau
    tau = 500;                                  % epoch where eta will be fully decayed

    % Early Stopping
    patience = 200;              
    tolerance = 0.0005;                         % minimum of 0.05% progress in patience epochs
    maxEpochs = 10000;
    
    %% ===================================
    % K-FOLD CROSS-VALIDATION
    % ====================================
    k = 5;
    cv = cvpartition(size(inputs_TR,1), 'KFold', k);
    
    input_layer_initial  = cell(1,k);
    hidden_layer_initial = cell(1,k);
    output_layer_initial = cell(1,k);

    rmse_train = nan(maxEpochs, k);
    rmse_val   = nan(maxEpochs, k);
    mee_val    = nan(maxEpochs, k);
    last_val_rmse = nan(1,k);

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

        best_val_mee = inf;
        epochs_since_improvement = 0;
        
        %% ===================================
        % NEURAL NETWORK CONFIGURATION (fully connected)
        % ====================================
        for i = 1:N
            input_layer(i) = neuron_input_unit(0);
        end
        
        for i = 1:numHidden
            hidden_layer(i) = neuron_hidden_unit(generate_hidden_conns_from(input_layer));
        end
        
        for i = 1:M
            output_layer(i) = neuron_output_unit(generate_output_conns_from(hidden_layer));
        end
        
        % Saving initial weights configuration
        input_layer_initial{fold}  = input_layer;
        hidden_layer_initial{fold} = hidden_layer;
        output_layer_initial{fold} = output_layer;

        %% ===================================
        % BACKPROPAGATION TRAINING LOOP
        % ====================================

        % Initialize velocity for weights and biases
        v_hidden_bias  = zeros(1, numHidden);
        v_hidden_weights = cell(1, numHidden);
        for j = 1:numHidden
            v_hidden_weights{j} = zeros(1,N);          % one per input -> hidden connection
        end
        
        v_output_bias = zeros(1,M);
        v_output_weights = cell(1,M);
        for m = 1:M
            v_output_weights{m} = zeros(1,numHidden);  % one per hidden -> output connection
        end

        epoch = 0;
        while epoch < maxEpochs
            epoch = epoch + 1;
        
            Yhat = zeros(P_train,M);

            % Shuffling patterns
            perm = randperm(P_train);
            A_train = A_train(perm,:);
            B_train_norm = B_train_norm(perm,:);
        
            % Loop over all patterns
            for p = 1:P_train
                %% Feedforward phase
                for i = 1:N
                    input_layer(i).output = A_train(p,i); % load pattern p
                end
                for j = 1:numHidden
                    hidden_layer(j).compute();
                end
                outputs = zeros(1,M);
                for m = 1:M
                    output_layer(m).compute();
                    outputs(m) = output_layer(m).output;
                end
                Yhat(p,:) = outputs;
        
                %% BackPropagation phase

                % Output signals
                output_signals = (outputs - B_train_norm(p,:));
        
                % Hidden layer signals
                hidden_signals = zeros(1,numHidden);
                for j = 1:numHidden
                    summation = 0;
                    for m = 1:M
                        summation = summation + output_signals(m) * output_layer(m).input_connections(j).weight;
                    end
                    hidden_signals(j) = summation * hidden_layer(j).Leaky_ReLU_derivative(hidden_layer(j).net);
                end
        
                %% Weights Update
                
                % Hidden -> Output
                for m = 1:M
                    % update bias velocity
                    v_output_bias(m) = alpha * v_output_bias(m) - eta_current * output_signals(m);
                    output_layer(m).bias_weight = output_layer(m).bias_weight + v_output_bias(m);
                
                    for j = 1:numHidden
                        % compute gradient
                        grad = output_signals(m) * hidden_layer(j).output;
                
                        % update velocity
                        v_output_weights{m}(j) = alpha * v_output_weights{m}(j) - eta_current * grad;
                
                        % update weight
                        output_layer(m).input_connections(j).weight = output_layer(m).input_connections(j).weight + v_output_weights{m}(j)...
                            - eta_current * lambda * sign(output_layer(m).input_connections(j).weight);
                    end
                end
        
                % Input -> Hidden
                for j = 1:numHidden
                    % update bias velocity
                    v_hidden_bias(j) = alpha * v_hidden_bias(j) - eta_current * hidden_signals(j);
                    hidden_layer(j).bias_weight = hidden_layer(j).bias_weight + v_hidden_bias(j);
                
                    for i = 1:N
                        % compute gradient
                        grad = hidden_signals(j) * input_layer(i).output;
                
                        % update velocity
                        v_hidden_weights{j}(i) = alpha * v_hidden_weights{j}(i) - eta_current * grad;
                
                        % update weight
                        hidden_layer(j).input_connections(i).weight = hidden_layer(j).input_connections(i).weight + v_hidden_weights{j}(i)...
                            - eta_current * lambda * sign(hidden_layer(j).input_connections(i).weight);
                    end
                end
            end
        
            %% Validation
            Yval = zeros(P_val,M);
            for p = 1:P_val
                for i = 1:N
                    input_layer(i).output = A_validation(p,i); % load pattern p
                end
                for j = 1:numHidden
                    hidden_layer(j).compute();
                end
                for m = 1:M
                    output_layer(m).compute();
                    Yval(p,m) = output_layer(m).output;
                end
            end
        
            err_tr = B_train_norm - Yhat;
            err_vl = B_val_norm   - Yval;
            
            rmse_per_output = sqrt(mean(err_tr.^2, 1));
            rmse_train(epoch,fold) = mean(rmse_per_output);
            rmse_val(epoch,fold)   = mean(sqrt(mean(err_vl.^2,1)));
            last_val_rmse(fold) = rmse_val(epoch, fold);
        
            % MEE computation
            Yval_den = Yval .* std_B + mu_B;
            diff_val = B_validation - Yval_den;
            mee_val(epoch,fold) = mean(sqrt(sum(diff_val.^2,2)));
        
            fprintf('MODEL -> h1 = %d; eta = %g; lambda = %g; alpha = %g\n',...
                numHidden, eta, lambda, alpha);
            fprintf('Epoch %d | Fold %d | RMSE_TR (norm) = %.6f | RMSE_VL (norm) = %.6f | MEE (og scale) = %.6f\n',...
                epoch, fold, rmse_train(epoch, fold), rmse_val(epoch, fold), mee_val(epoch, fold));
            disp(rmse_per_output);
        
            % If it's diverging, stop
            if isnan(mee_val(epoch,fold)) || isinf(mee_val(epoch,fold))
                fprintf('MODEL -> h1 = %d; eta = %g; lambda = %g; alpha = %g\n',...
                    numHidden, eta, lambda, alpha);
                fprintf("NaN detected at epoch %d fold %d — stopping model\n\n", epoch, fold);
                break;
            end
        
            if epoch > tau
                % Early Stopping based on MEE (og scale)
                if mee_val(epoch,fold) < best_val_mee * (1 - tolerance)
                    best_val_mee = mee_val(epoch,fold);
                    epochs_since_improvement = 0;
                else
                    epochs_since_improvement = epochs_since_improvement + 1;
                end
            
                if epochs_since_improvement >= patience
                    fprintf("EARLY STOP at epoch %d | Best MEE = %.6f\n\n", epoch, best_val_mee);
                    break;
                end
            else
                % before tau, always reset patience
                best_val_mee = min(best_val_mee, mee_val(epoch,fold));
                epochs_since_improvement = 0;
            end

            % Variable (linear decaying) learning rate
            if epoch <= tau
                gamma = epoch / tau;
                eta_current = (1 - gamma) * eta + gamma * eta_tau;
            else
                eta_current = eta_tau;
            end
        end
        input_layer_final{fold}  = input_layer;
        hidden_layer_final{fold} = hidden_layer;
        output_layer_final{fold} = output_layer;
    end

    training_end_time = posixtime(datetime('now'));

    % Saving model's data
    model.rmse_min = min(nanmean(rmse_train,2));
    model.rmse_final = nanmean(rmse_train(end,:));
    model.rmse_validation = nanmean(last_val_rmse);
    model.mee_min = min(nanmean(mee_val,2));
    model.mee_final = nanmean(mee_val(end,:));

    model.eta = eta;
    model.alpha = alpha;
    model.lambda = lambda;
    model.numHidden = numHidden;

    model.input_layer_initial  = input_layer_initial;
    model.hidden_layer_initial = hidden_layer_initial;
    model.output_layer_initial = output_layer_initial;

    model.input_layer_final  = input_layer_final;
    model.hidden_layer_final = hidden_layer_final;
    model.output_layer_final = output_layer_final;

    model.training_time = training_end_time - training_start_time;

    if ~exist('models', 'dir')
        mkdir('models');
    end

    filename = sprintf( ...
        'models/h1-%d-eta-%g-lambda-%g-alpha-%g_%d.mat', ...
        numHidden, eta, lambda, alpha, randi(1e6));
    save(filename, 'model');

    % Plotting and saving the learning curve
    mean_train_curve = nanmean(rmse_train,2);
    mean_val_curve   = nanmean(rmse_val,2);

    max_trained_epoch = max(sum(~isnan(rmse_train),1));

    [~, name, ~] = fileparts(filename);
    plot_file = fullfile('models', [name '_plot.png']);
    
    fig = figure('Visible','off');
    plot(1:max_trained_epoch, mean_train_curve(1:max_trained_epoch), 'b', 'LineWidth', 2); hold on;
    plot(1:max_trained_epoch, mean_val_curve(1:max_trained_epoch), 'r', 'LineWidth', 2);
    xlabel('Epoch'); ylabel('RMSE');
    title(sprintf('Learning Curve |  h1 = %d; eta = %g; lambda = %g; alpha = %g; SGD', ...
        numHidden, eta, lambda, alpha));
    grid on;
    
    exportgraphics(fig, plot_file);
    close(fig);

    score = min(nanmean(mee_val,2));
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
