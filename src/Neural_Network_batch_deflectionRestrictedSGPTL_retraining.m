function score = Neural_Network_batch_deflectionRestrictedSGPTL_retraining(numHidden1, numHidden2, lambda, beta, delta_multiplicator, R, rho)
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
    % lambda                                    % factor for L1 Regularization

    % Early Stopping
    maxEpochs = 10000;
    rmse_train = nan(maxEpochs);

    training_start_time = posixtime(datetime('now'));
    
    P_train = size(inputs_TR,1);
    
    mee_history = nan(1, maxEpochs);
    rmse_history = nan(1, maxEpochs);

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
    % I/O NORMALIZATION (zero-mean / unit-variance)
    % ====================================

    mu_A  = mean(inputs_TR,1);
    std_A = std(inputs_TR,0,1);
    std_A = max(std_A, 1e-8);

    A_train = (inputs_TR - mu_A) ./ std_A;

    mu_B  = mean(outputs_TR,1);
    std_B = std(outputs_TR,0,1);
    std_B = max(std_B, 1e-8);

    B_train_norm = (outputs_TR - mu_B) ./ std_B;

    %% ===================================
    % BACKPROPAGATION TRAINING LOOP
    % ====================================

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
            denorm_diff = outputs_TR(p,:) - outputs_denorm;
            total_error = total_error + sqrt(sum(denorm_diff.^2));
            %% BackPropagation phase
        
            % Output signals
            output_signals = zeros(1,M);
            for s = 1:M
                output_signals(s) = (B_train_norm(p,s) - outputs(s));
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
            for s = 1:M
                grad_b_out(s) = grad_b_out(s) + output_signals(s);
                for j = 1:numHidden2
                    grad_W_out(s,j) = grad_W_out(s,j) + output_signals(s) * hidden_layer2(j).output;
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
        mse_loss =  mean((B_train_norm - Yhat).^2, 'all');

            L1 = 0;

            for j = 1:numHidden1
                for i = 1:N
                    L1 = L1 + abs(hidden_layer1(j).input_connections(i).weight);
                end
            end
            
            for j = 1:numHidden2
                for i = 1:numHidden1
                    L1 = L1 + abs(hidden_layer2(j).input_connections(i).weight);
                end
            end
            
            for s = 1:M
                for j = 1:numHidden2
                    L1 = L1 + abs(output_layer(s).input_connections(j).weight);
                end
            end

            W_h1 = zeros(numHidden1, N);
            for j = 1:numHidden1
                for i = 1:N
                    W_h1(j,i) = hidden_layer1(j).input_connections(i).weight;
                end
            end

            W_h2 = zeros(numHidden2, numHidden1);
            for j = 1:numHidden2
                for i = 1:numHidden1
                    W_h2(j,i) = hidden_layer2(j).input_connections(i).weight;
                end
            end

            W_out = zeros(M, numHidden2);
            for j = 1:M
                for i = 1:numHidden2
                    W_out(j,i) = output_layer(j).input_connections(i).weight;
                end
            end

            num_weights = numel(W_h1) + numel(W_h2) + numel(W_out);
            loss = mse_loss + lambda * (L1 / num_weights);

            g = [
                grad_W_h1(:) + lambda * sign(W_h1(:));
                grad_b_h1(:);
                grad_W_h2(:) + lambda * sign(W_h2(:));
                grad_b_h2(:);
                grad_W_out(:) + lambda * sign(W_out(:));
                grad_b_out(:)
            ] / P_train;

    
            if epoch == 1
                gamma = 1;
                d_prev = g;
                f_ref =  loss;
                f_best =  loss;
                delta = delta_multiplicator * loss;
                r = 0;
            else
                num = dot(d_prev, d_prev - g);
                den = norm(g - d_prev)^2 + 1e-8; 
                gamma_star = num / den;
                if ~isfinite(num) || ~isfinite(den)
                    gamma = 0;
                else
                    gamma = min(1, max(0, gamma_star));
                    %fprintf('GammaStar: %f ', gamma_star)
                end

                if loss > f_best && alpha_prev > 0
                    gammaRestriction = (alpha_prev * norm(d_prev)^2) / ...
                    ((loss - f_best) + (alpha_prev * norm(d_prev)^2) + 1e-12);
                    gamma = min(gamma, max(0, gammaRestriction));     
                end

            end
    
            d_vec = gamma * g + (1 - gamma) * d_prev;
            d = (norm(d_vec)^2);
    
            numeratore = max(0, loss - (f_ref - delta));

            epsilon = 1e-8;
            alpha = beta * numeratore / (d + epsilon);
            alpha = max(alpha, 1e-6);
            alpha = min(alpha, 0.1);
        
            if(loss <= f_ref-delta/2) %significa che si è arrivati vicini al valore ottimo stimato quindi si migliora
                f_ref = f_best;
                r = 0;
            elseif(r > R) %significa che mi sono mosso troppo senza miglioramenti significativi
                delta = delta * rho; %aggiorno delta con il fattore rho 0<rho<1 per cercare valori meno ambiziosi
                r = 0;
            else
                r = r + alpha * sqrt(d); %aggiorno con la distanza percorsa a questa iterazione
            end
            f_best = min(f_best,  loss);
            d_prev = d_vec;
            alpha_prev = alpha;
        %fprintf('Loss: %f | Alpha: %.8f | Gamma: %.8f | f_best: %.8f | r:%f | R:%f | d:%f\n', loss, alpha, gamma, f_best, r, R, d);
    
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
                    hidden_layer1(j).input_connections(i).weight + ...
                        alpha * (d_W_h1(j,i));
            end
        end
        
        % Hidden1 -> Hidden2
        for j = 1:numHidden2
            hidden_layer2(j).bias_weight = hidden_layer2(j).bias_weight + alpha * d_b_h2(j);
            for i = 1:numHidden1
                hidden_layer2(j).input_connections(i).weight = ...
                    hidden_layer2(j).input_connections(i).weight  ...
                       + alpha *  (d_W_h2(j,i));
            end
        end
        
        % Hidden2 -> Output
        for s = 1:M
            output_layer(s).bias_weight = output_layer(s).bias_weight + alpha * d_b_out(s);
            for j = 1:numHidden2
                output_layer(s).input_connections(j).weight = ...
                    output_layer(s).input_connections(j).weight + ...
                       alpha * (d_W_out(s,j));
            end
        end
        
        % RMSE with collected network outputs over an epoch
        err = B_train_norm - Yhat;
        rmse_per_output = sqrt(mean(err.^2, 1));
        rmse_history(epoch) = mean(rmse_per_output);
        rmse_train(epoch) = rmse_history(epoch);
        
        % Compute Mean Euclidian Error over an epoch
        mee_history(epoch) = total_error / P_train;
        %fprintf('Epoch %d | RMSE (norm) = %.6f | MEE (og scale) = %.6f\n', epoch, rmse_history(epoch), mee_history(epoch));

   end

    training_end_time = posixtime(datetime('now'));
    input_layer_final  = input_layer;
    hidden_layer1_final = hidden_layer1;
    hidden_layer2_final = hidden_layer2;
    output_layer_final  = output_layer;

    % Saving model's data
    model.rmse_train = min(nanmean(rmse_train,2));

    model.lambda = lambda;
    model.numHidden1 = numHidden1;
    model.numHidden2 = numHidden2;
    model.beta = beta;
    model.delta = delta_multiplicator;
    model.R = R;
    model.rho = rho;

    model.input_layer_final = input_layer_final;
    model.hidden_layer1_final = hidden_layer1_final;
    model.hidden_layer2_final = hidden_layer2_final;
    model.output_layer_final = output_layer_final;

    model.mu_A  = mu_A;
    model.std_A = std_A;
    model.mu_B  = mu_B;
    model.std_B = std_B;

    model.training_time = training_end_time - training_start_time;

    if ~exist('models', 'dir')
        mkdir('models');
    end

    filename = sprintf( ...
        'models/SGPTL/deflectionRestrictedSGPTL-h1-%d-h2-%d-lambda-%g-beta-%g-R-%g-rho-%g-deltaMult-%g_%d.mat', ...
        numHidden1, numHidden2, lambda, beta, R, rho, delta_multiplicator, randi(1e6));
    save(filename, 'model');

    score = 1;

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
            fan_in = numel(hidden_units);
            fan_out = 4; % Numero di output (M)
            
            % Deviazione standard Xavier per layer lineare
            xavier_std = sqrt(2 / (fan_in + fan_out)); 
        
            output_conns(1, fan_in) = struct('neuron',[],'weight',[]);
            for n = 1:fan_in
                output_conns(n).neuron = hidden_units(n);
                % randn garantisce pesi distribuiti intorno allo zero con la giusta varianza
                output_conns(n).weight = randn * xavier_std; 
            end
        end
    end
