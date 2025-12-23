function score = Neural_Network_batch_deflection_restricted(numHidden1, numHidden2, lambda, beta, delta_multiplicator)
    %% ===================================
    % LOADING TRAINING DATA (500 patterns)
    % ====================================
    Dataset_TR = readtable('../data/TR/ML-CUP25-TR.csv');
    
    inputs_TR  = Dataset_TR{:, 2:13};
    outputs_TR = Dataset_TR{:, 14:end};
    
    N = size(inputs_TR, 2);                     % 12 inputs
    M = size(outputs_TR, 2);                    % 4 outputs
    
    % Hyper Parameters
    % numHidden1                                % # of units inside first Hidden Layer
    % numHidden2                                % # of units inside second Hidden Layer
    
    % eta                                       % initial Learning Rate
    %eta_tau = eta / 100;                       % eta to use from epoch tau
    %tau = 200;                                 % epoch where eta will be fully decayed
    
    % lambda                                    % factor for L1 Regularization

    % Early Stopping
    patience = 200;              
    tolerance = 0.05;                           % minimum of 5% progress in patience epochs
    maxEpochs = 10000;
    
    %% ===================================
    % NEURAL NETWORK CONFIGURATION (fully connected)
    % ====================================
    
    % Input Layer
    for i = 1:N
        input_layer(i) = neuron_input_unit(0);
    end
    
    % Hidden Layer 1
    for i = 1:numHidden1
        hidden_layer1(i) = neuron_hidden_unit(generate_hidden_conns_from(input_layer));
    end
    
    % Hidden Layer 2
    for i = 1:numHidden2
        hidden_layer2(i) = neuron_hidden_unit(generate_hidden_conns_from(hidden_layer1));
    end
    
    % Output Layer
    for i = 1:M
        output_layer(i) = neuron_output_unit(generate_output_conns_from(hidden_layer2));
    end

    % Saving initial weights configuration
    input_layer_initial = input_layer;
    hidden_layer1_initial = hidden_layer1;
    hidden_layer2_initial = hidden_layer2;
    output_layer_initial = output_layer;
    
    %% ===================================
    % K-FOLD CROSS-VALIDATION
    % ====================================
    k = 5;
    cv = cvpartition(size(inputs_TR,1), 'KFold', k);

    training_start_time = posixtime(datetime('now'));

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
        % BACKPROPAGATION TRAINING LOOP
        % ====================================
    
        mee_history = zeros(1, maxEpochs);
        rmse_history = zeros(1, maxEpochs);

        epoch = 0;
        while epoch < maxEpochs
            epoch = epoch + 1;
        
            total_error = 0;
            Yhat = zeros(P_train, M);
    
            % Gradient Accumulators
            grad_W_h1 = zeros(numHidden1, N); grad_b_h1 = zeros(1,numHidden1);
            grad_W_h2 = zeros(numHidden2, numHidden1); grad_b_h2 = zeros(1,numHidden2);
            grad_W_out = zeros(M,numHidden2); grad_b_out = zeros(1,M);

            % Loop over all patterns
            for p = 1:P_train
                %% Feedforward phase
                for i = 1:N
                    input_layer(i).output = A_train(p,i); % load pattern p
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
                
                outputs_denorm = outputs .* std_B + mu_B;
                denorm_diff = B_train(p,:) - outputs_denorm;
                total_error = total_error + sqrt(sum(denorm_diff.^2));
                %% BackPropagation phase
        
                % Output signals
                output_signals = zeros(1,M);
                for k = 1:M
                    output_signals(k) = (B_train_norm(p,k) - outputs(k));
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
            
            %% Validation

            Yval = zeros(P_val,M);
            for p = 1:P_val
                for i = 1:N
                    input_layer(i).output = A_validation(p,i); % load pattern p
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
                    Yval(p,:) = outputs;
                end
            end

            err_val = B_val_norm - Yval;
            rmse_val(epoch, fold) = sqrt(mean(err_val(:).^2));
        
            %% Weights Update
            
            loss = sum(sum((B_train_norm - Yhat).^2)) / P_train;
            g = [
                grad_W_h1(:);
                grad_b_h1(:);
                grad_W_h2(:);
                grad_b_h2(:);
                grad_W_out(:);
                grad_b_out(:)
            ] / P_train; %metto i gradienti in un unico vettore
    
            if epoch > 200
                delta = loss * delta_multiplicator * 1e-1;
            else
                delta = loss * delta_multiplicator;
            end
            
            if epoch == 1
                gamma = 1;
                d_prev = g;
                %delta = delta * loss;
                f_best = loss - delta;
            else
                g_norm = g / (norm(g) + 1e-12);
                d_prev_norm = d_prev / (norm(d_prev) + 1e-12);
    
                num = dot(d_prev_norm, d_prev_norm - g_norm);
                den = norm(g_norm - d_prev_norm)^2 + 1e-6; % 1e-6 agisce da smorzatore
                gamma_star = num / den;
                gamma = min(1, max(0, gamma_star));
                fprintf('GammaStar: %f ', gamma_star)
                if loss <= f_best
                    f_best = loss - delta; % Aggiornamento standard
                else
                % Se la loss sale o stagna, f_best è troppo ottimistico
                    f_best = f_best + 0.5 * delta; 
                end
                gammaRestriction = (alpha_prev * norm(d_prev)^2)/((loss - f_best)+(alpha_prev*norm(d_prev)^2));
                gamma = min(gamma, gammaRestriction);
            end
            %per combattere vanishing
            if gamma < 0.05
                d_vec = g; % Ignora la restrizione per un passo per "riattivare" il sistema
            else
                d_vec = gamma * g + (1 - gamma) * d_prev;
            end
    
            epsilon = 1e-8;
            alpha = beta * (loss - f_best) / (norm(d_vec)^2 + epsilon);
    
            alpha = max(alpha, 0);   %per evitare alpha negativi
            alpha_prev = alpha;
            fprintf('Loss: %f | Alpha: %.8f | Gamma: %.8f | f_best: %.8f\n', loss, alpha, gamma, f_best);
    
            % Calcola gli indici di slicing
            idx1 = 1 : numHidden1*N;                   
            idx2 = idx1(end)+1 : idx1(end)+numHidden1; 
            idx3 = idx2(end)+1 : idx2(end)+numHidden2*numHidden1; 
            idx4 = idx3(end)+1 : idx3(end)+numHidden2; 
            idx5 = idx4(end)+1 : idx4(end)+M*numHidden2;         
            idx6 = idx5(end)+1 : idx5(end)+M;
    
            % Estrai le porzioni da d_vec
            d_W_h1   = reshape(d_vec(idx1), numHidden1, N);
            d_b_h1   = reshape(d_vec(idx2), numHidden1, 1);
            d_W_h2   = reshape(d_vec(idx3), numHidden2, numHidden1);
            d_b_h2   = reshape(d_vec(idx4), numHidden2, 1);
            d_W_out  = reshape(d_vec(idx5), M, numHidden2);
            d_b_out  = reshape(d_vec(idx6), M, 1);
        
            % Input -> Hidden1
            for j = 1:numHidden1
                hidden_layer1(j).bias_weight = hidden_layer1(j).bias_weight + alpha * d_b_h1(j);
                for i = 1:N
                    hidden_layer1(j).input_connections(i).weight = ...
                        hidden_layer1(j).input_connections(i).weight + alpha * d_W_h1(j,i) - ...
                            lambda * sign(hidden_layer1(j).input_connections(i).weight);
                end
            end
        
            % Hidden1 -> Hidden2
            for j = 1:numHidden2
                hidden_layer2(j).bias_weight = hidden_layer2(j).bias_weight + alpha * d_b_h2(j) / P_train;
                for i = 1:numHidden1
                    hidden_layer2(j).input_connections(i).weight = ...
                        hidden_layer2(j).input_connections(i).weight + alpha *  d_W_h2(j,i) / P_train - ...
                            lambda * sign(hidden_layer2(j).input_connections(i).weight);
                end
            end
        
            % Hidden2 -> Output
            for k = 1:M
                output_layer(k).bias_weight = output_layer(k).bias_weight + alpha * d_b_out(k) / P_train;
                for j = 1:numHidden2
                    output_layer(k).input_connections(j).weight = ...
                        output_layer(k).input_connections(j).weight + alpha * d_W_out(k,j) / P_train - ...
                            lambda * sign(output_layer(k).input_connections(j).weight);
                end
            end
        
            % RMSE with collected network outputs over an epoch
            err = B_train_norm - Yhat;
            rmse_per_output = sqrt(mean(err.^2, 1));
        
            % Compute Mean Euclidian Error over an epoch
            mee_history(epoch) = total_error / P_train;
            rmse_history(epoch) = mean(rmse_per_output);
            rmse_train(epoch, fold) = rmse_history(epoch);

            fprintf('Epoch %d | RMSE (norm) = %.6f | MEE (og scale) = %.6f | Alpha = %.6f\n', epoch, rmse_history(epoch), mee_history(epoch), alpha);
            disp(rmse_per_output);
        
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
            d_prev = d_vec;
        end
    end

    training_end_time = posixtime(datetime('now'));

    % Saving model's data
    model.rmse_min = min(mean(rmse_train,2));
    model.rmse_final = mean(rmse_train(end,:));
    model.rmse_validation = mean(rmse_val(end,:));
    model.mee_min = min(mee_history);
    model.mee_final = mee_history(end);

    model.alpha = alpha;
    model.lambda = lambda;
    model.numHidden1 = numHidden1;
    model.numHidden2 = numHidden2;

    model.input_layer_initial = input_layer_initial;
    model.hidden_layer1_initial = hidden_layer1_initial;
    model.hidden_layer2_initial = hidden_layer2_initial;
    model.output_layer_initial = output_layer_initial;

    model.input_layer_final = input_layer;
    model.hidden_layer1_final = hidden_layer1;
    model.hidden_layer2_final = hidden_layer2;
    model.output_layer_final = output_layer;

    model.training_time = training_end_time - training_start_time;

    if ~exist('models', 'dir')
        mkdir('models');
    end

    filename = sprintf( ...
        'models/deflectionRestricted-h1-%d-h2-%d-lambda-%g-beta-%g-deltaMult-%g_%d.mat', ...
        numHidden1, numHidden2, lambda, beta, delta_multiplicator, randi(1e6));
    save(filename, 'model');

    % Plotting and saving the learning curve
    [~, name, ~] = fileparts(filename);
    plot_file = fullfile('models', [name '_plot.png']);
    
    fig = figure('Visible','off');
    plot(1:size(rmse_train, 2), mean(rmse_train, 2), 'b', 'LineWidth', 2); hold on;
    plot(1:size(rmse_val, 2), mean(rmse_val, 2), 'r', 'LineWidth', 2);
    xlabel('Epoch'); ylabel('RMSE');
    title(sprintf('Learning Curve | h1 = %d; h2 = %d; lambda = %g; beta = %g; deltaMult = %g; Batch', ...
        numHidden1, numHidden2, lambda, beta, delta_multiplicator));
    grid on;
    
    exportgraphics(fig, plot_file);
    close(fig);

    score = mee_history(end);
%%
    function hidden_conns = generate_hidden_conns_from(input_units)
        % Kaiming Initialization
        fan_in = numel(input_units);
        kaiming = sqrt(2 / fan_in);

        hidden_conns(1, fan_in) = struct('neuron',[],'weight',[]);
        for n = 1:fan_in
            hidden_conns(n).neuron = input_units(n);
            hidden_conns(n).weight = randn * kaiming;
        end
    end
    
    function output_conns = generate_output_conns_from(hidden_units)
        % Kaiming Initialization
        fan_in = numel(hidden_units);
        kaiming = sqrt(2 / fan_in);
    
        output_conns(1, fan_in) = struct('neuron',[],'weight',[]);
        for n = 1:fan_in
            output_conns(n).neuron = hidden_units(n);
            output_conns(n).weight = randn * kaiming;
        end
    end
end
