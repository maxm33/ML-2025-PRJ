function score = Neural_Network_batch_stepsize_restricted(numHidden1, numHidden2, lambda, beta, delta_multiplicator)
    %% ===================================
    % LOADING TRAINING DATA (500 patterns)
    % ====================================
    Dataset_TR = readtable('../data/TR/ML-CUP25-TR.csv');
    
    inputs_TR  = Dataset_TR{:, 2:13};
    outputs_TR = Dataset_TR{:, 14:end};
    
    N = size(inputs_TR, 2);                     % 12 inputs
    M = size(outputs_TR, 2);                    % 4 outputs
    %P = size(inputs_TR, 1);                     % 500 patterns
    
    % Hyper Parameters
    % numHidden1                                % # of units inside first Hidden Layer
    % numHidden2                                % # of units inside second Hidden Layer
    
    % eta                                       % initial Learning Rate
    %eta_tau = 0.001;                           % eta to use from epoch tau
    %tau = 200;                                 % epoch where eta will be fully decayed
    
    % lambda                                    % factor for L1 Regularization
    % Early Stopping
    patience = 200;              
    tolerance = 1e-2;  

    epochs = 1000;
    
    %A = (inputs_TR - mean(inputs_TR,1)) ./ std(inputs_TR,0,1);
    %B = (outputs_TR - mean(outputs_TR,1)) ./ std(outputs_TR,0,1);

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
        
        %% ===================================
        % BACKPROPAGATION TRAINING LOOP
        % ====================================
    
        mee_history = zeros(1, epochs);
        rmse_history = zeros(1, epochs);
        epoch_times = zeros(1, epochs);
        
        % Plot Initialization
        %figure;
        %hLine = plot(NaN, NaN, 'b-', 'LineWidth', 2);
        %xlabel('Epoch'); ylabel('RMSE'); title('Learning Curve'); grid on; hold on;

        for epoch = 1:epochs
            epoch_time_start = posixtime(datetime('now'));
        
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
    
            if epoch > 200 %la discesa inizia a essere più lenta
                delta = loss * delta_multiplicator * 1e-1;
            else
                delta = loss * delta_multiplicator;
            end
    
            if epoch == 1
                gamma = 1;
                d_prev = g;
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
                % Se la loss sale o stagna, f_best è troppo ottimistico e
                % quindi dimezzo il fattore delta
                    f_best = f_best + 0.5 * delta; 
                end
            end
    
           %per combattere vanishing
            if gamma < 0.05
                d_vec = g; % Ignora la restrizione per un passo per "riattivare" il sistema
                beta_eff = beta;
            else
                d_vec = gamma * g + (1 - gamma) * d_prev;
                beta_eff = min(beta, gamma);%condizione di stepsize restricted
            end
    
            epsilon = 1e-8;
            alpha = beta_eff * (loss - f_best) / (norm(d_vec)^2 + epsilon);
    
            alpha = max(alpha, 0);   %per evitare alpha negativi
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
                improvement = rmse_history(epoch - patience) - rmse_history(epoch);
                
                if improvement < tolerance && rmse_history(epoch) > 0.5
                    fprintf('EARLY STOPPING: improvement < %.4f over last %d epochs\n', ...
                            tolerance, patience);
                    score = 200;
                    return;
                end
            end

            d_prev = d_vec;
            epoch_times(epoch) = posixtime(datetime('now')) - epoch_time_start;
        end
        
    end
    fprintf('Total Training Time (seconds) = %.3f | Average Epoch Time (seconds) = %.3f\n', sum(epoch_times), mean(epoch_times));

    % Saving model's data
    model.rmse_final =  mean(rmse_train(end,:));
    model.rmse_min = min(mean(rmse_train,2));
    model.mee_final = mee_history(end);
    model.mee_min = min(mee_history);
    model.rmse_validation = mean(rmse_val(end,:));

    model.input_layer = input_layer;
    model.hidden_layer1 = hidden_layer1;
    model.hidden_layer2 = hidden_layer2;
    model.output_layer = output_layer;

    model.numHidden1 = numHidden1;
    model.numHidden2 = numHidden2;
    model.lambda = lambda;
    model.alpha = alpha;
    
    model.mu_input = mean(inputs_TR,1);
    model.sigma_input = std(inputs_TR,0,1);
    model.mu_output = mean(outputs_TR,1);
    model.sigma_output = std(outputs_TR,0,1);

    if ~exist('models', 'dir')
        mkdir('models');
    end

    filename = sprintf( ...
        'models/stepsizeRestricted-h1-%d-h2-%d-lambda-%g-beta-%g-deltaMult-%g_%d.mat', ...
        numHidden1, numHidden2, lambda, beta, delta_multiplicator, randi(1e6));
    save(filename, 'model');

    rmse_train_mean = mean(rmse_train, 2);
    rmse_val_mean   = mean(rmse_val, 2);

    % Plotting and saving the learning curve
    [~, name, ~] = fileparts(filename);
    plot_file = fullfile('models', [name '_plot.png']);
    
    fig = figure('Visible','off');
    plot(1:epoch, rmse_train_mean, 'b', 'LineWidth', 2); hold on;
    plot(1:epoch, rmse_val_mean, 'r', 'LineWidth', 2);
    xlabel('Epoch'); ylabel('RMSE');
    title(sprintf('Learning Curve | h1 = %d; h2 = %d; lambda = %g; beta = %g; deltaMult = %g; Batch', ...
        numHidden1, numHidden2, lambda, beta, delta_multiplicator));
    grid on;
    
    exportgraphics(fig, plot_file);
    close(fig);

    score = mee_history(end);
    
    %%
    function hidden_conns = generate_hidden_conns_from(input_units)
        hidden_conns(1, numel(input_units)) = struct('neuron',[],'weight',[]);
    
        for n = 1:numel(input_units)
            hidden_conns(n).neuron = input_units(n);
            hidden_conns(n).weight = randn * 0.1;
        end
    end
    
    function output_conns = generate_output_conns_from(hidden_units)
        output_conns(1, numel(hidden_units)) = struct('neuron',[],'weight',[]);
    
        for n = 1:numel(hidden_units)
            output_conns(n).neuron = hidden_units(n);
            output_conns(n).weight = randn * 0.1;
        end
    end
end
