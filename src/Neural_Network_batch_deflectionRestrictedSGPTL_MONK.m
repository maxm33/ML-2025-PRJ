function score = Neural_Network_batch_deflectionRestrictedSGPTL_MONK(numHidden1, numHidden2, lambda, beta, delta_multiplicator, R, rho)
    %% ===================================
    % LOADING TRAINING DATA (500 patterns)
    % ====================================
    % per MONK
    Dataset_TR = readtable('../monk/monks-3.train', ...
    'FileType','text', ...
    'Delimiter',' ', ...
    'MultipleDelimsAsOne',true, ...
    'ReadVariableNames',false, ...
    'TextType','string');

    % rimuove colonne vuote
    Dataset_TR(:, all(ismissing(Dataset_TR),1)) = [];
    
    X  = Dataset_TR{:, 2:7};
    Y = Dataset_TR{:, 1};

    X_oh = [];
    cats = cell(1, size(X,2));
    
    for j = 1:size(X,2)
        cats{j} = categories(categorical(X(:,j)));
        X_cat_j = categorical(X(:,j), cats{j});
        X_oh = [X_oh dummyvar(X_cat_j)];
    end

    inputs_TR  = X_oh;
    outputs_TR = Y;
    outputs_TR = double(outputs_TR > 0);
    
    N = size(inputs_TR, 2);                     
    M = 1;

    %% Carico test set
    Dataset_TS = readtable('../monk/monks-3.test', ...
    'FileType','text', ...
    'Delimiter',' ', ...
    'MultipleDelimsAsOne',true, ...
    'ReadVariableNames',false, ...
    'TextType','string');

    Dataset_TS(:, all(ismissing(Dataset_TS),1)) = [];

    X_test = Dataset_TS{:, 2:7};
    Y_test = double(Dataset_TS{:, 1} > 0);

    X_oh_test = [];
    for j = 1:size(X_test,2)
        X_test_cat_j = categorical(X_test(:,j), cats{j});
        X_oh_test = [X_oh_test dummyvar(X_test_cat_j)];
    end
    
    % Early Stopping
    %patience = 300;                                       
    maxEpochs = 2000;
    
    %% ===================================
    % K-FOLD CROSS-VALIDATION
    % ====================================
    k = 5;
    cv = cvpartition(size(inputs_TR,1), 'KFold', k);

    input_layer_initial  = cell(1,k);
    hidden_layer1_initial = cell(1,k);
    % hidden_layer2_initial = cell(1,k);
    output_layer_initial  = cell(1,k);

    acc_val = nan(maxEpochs, k);
    acc_train = nan(maxEpochs, k);
    mse_val = nan(maxEpochs, k);
    mse_train = nan(maxEpochs, k);
    best_val_acc = nan(k,1);
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

        best_val_acc(fold) = 0;
        %epochs_since_improvement = 0;

        %% ===================================
        % NEURAL NETWORK CONFIGURATION (fully connected)
        % ====================================
        
        % Input Layer
        for i = 1:N
            input_layer(i) = neuron_input_unit(0);
        end
        
        % Hidden Layer 1
        for i = 1:numHidden1
            hidden_layer1(i) = neuron_hidden_unit_tahn(generate_hidden_conns_from(input_layer, numHidden1));
        end
        
        % Hidden Layer 2
        % for i = 1:numHidden2
        %     hidden_layer2(i) = neuron_hidden_unit_tahn(generate_hidden_conns_from(hidden_layer1, numHidden2));
        % end
        
        % Output Layer
        for i = 1:M
            output_layer(i) = neuron_output_unit_sigmoid(generate_output_conns_from(hidden_layer1));
        end
    
        % Saving initial weights configuration
        input_layer_initial{fold}  = input_layer;
        hidden_layer1_initial{fold} = hidden_layer1;
        output_layer_initial{fold}  = output_layer;

        %% ===================================
        % I/O NORMALIZATION (zero-mean / unit-variance)
        % ====================================
        A_train = inputs_TR(train_idx,:);
        A_validation   = inputs_TR(validation_idx,:);

        %% ===================================
        % BACKPROPAGATION TRAINING LOOP
        % ====================================

        epoch = 0;
        while epoch < maxEpochs
            epoch = epoch + 1;
            Yhat = zeros(P_train, M);
    
            % Gradient Accumulators
            grad_W_h1 = zeros(numHidden1, N); grad_b_h1 = zeros(1,numHidden1);
            %grad_W_h2 = zeros(numHidden2, numHidden1); grad_b_h2 = zeros(1,numHidden2);
            grad_W_out = zeros(M,numHidden1); grad_b_out = zeros(1,M);

            % Loop over all patterns
            for p = 1:P_train
                %% Feedforward phase
                for i = 1:N
                    input_layer(i).output = A_train(p,i); % load pattern p
                end
                for i = 1:numHidden1
                    hidden_layer1(i).compute();
                end
                % for i = 1:numHidden2
                %     hidden_layer2(i).compute();
                % end
                outputs = zeros(1,M);
                for i = 1:M
                    output_layer(i).compute();
                    outputs(i) = output_layer(i).output;
                    Yhat(p,:) = outputs;
                end

                %% BackPropagation phase
        
                % Output signals
                output_signals = zeros(1,M);
                for k = 1:M
                    output_signals(k) = (B_train(p,k) - outputs(k));
                end
        
                % Hidden layer 2 signals
                % hidden2_signals = zeros(1, numHidden2);
                % for j = 1:numHidden2
                %     summation = 0;
                %     for k = 1:M
                %         summation = summation + output_signals(k) * output_layer(k).input_connections(j).weight;
                %     end
                %     hidden2_signals(j) = summation * hidden_layer2(j).activation_derivative(hidden_layer2(j).net);
                % end
        
                % Hidden layer 1 signals
                hidden1_signals = zeros(1,numHidden1);
                for j = 1:numHidden1
                    summation = 0;
                    for k = 1:M
                        summation = summation + output_signals(k) * output_layer(k).input_connections(j).weight;
                    end
                    hidden1_signals(j) = summation * hidden_layer1(j).activation_derivative(hidden_layer1(j).net);
                end

        
                %% Gradients Accumulation
        
                % Output layer
                for k = 1:M
                    grad_b_out(k) = grad_b_out(k) + output_signals(k);
                    for j = 1:numHidden1
                        grad_W_out(k,j) = grad_W_out(k,j) + output_signals(k) * hidden_layer1(j).output;
                    end
                end
        
                % Hidden layer 2
                % for j = 1:numHidden2
                %     grad_b_h2(j) = grad_b_h2(j) + hidden2_signals(j);
                %     for i = 1:numHidden1
                %         grad_W_h2(j,i) = grad_W_h2(j,i) + hidden2_signals(j) * hidden_layer1(i).output;
                %     end
                % end
        
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
                outputs = zeros(1,M);
                for i = 1:M
                    output_layer(i).compute();
                    outputs(i) = output_layer(i).output;
                    Yval(p,:) = outputs;
                end
            end
            
            mse_train(epoch, fold) = mean((B_train - Yhat).^2,'all');
            mse_val(epoch, fold)   = mean((B_validation - Yval).^2,'all');

            %% Calcolo dell'accuracy
            Yhat_bin = Yhat > 0.5;
            acc_train(epoch, fold) = mean(Yhat_bin == B_train);
            Yval_bin = Yval > 0.5;
            acc_val(epoch, fold) = mean(Yval_bin == B_validation);

            %% Weights Update
            
            loss = mean((B_train - Yhat).^2,'all');

            g = [
                grad_W_h1(:);
                grad_b_h1(:);
                grad_W_out(:);
                grad_b_out(:)
            ] / P_train; %metto i gradienti in un unico vettore
    
            if epoch == 1
                gamma = 1;
                d_prev = g;
                f_ref =  loss;
                f_best =  loss;
                delta = delta_multiplicator * loss;
                r = 0;
            else
                diff = g - d_prev;
                num = dot(d_prev, diff);
                den = dot(diff, diff) + 1e-8;

                gamma_star = 1 - num / den;
                gamma = min(1, max(0, gamma_star));

                if loss > f_best && alpha_prev > 0
                    gammaRestriction = (alpha_prev * norm(d_prev)^2) / ...
                    ((loss - f_best) + (alpha_prev * norm(d_prev)^2) + 1e-12);
                    gamma = min(gamma, max(0, gammaRestriction));
                end

            end
    
            d_vec = gamma * g + (1 - gamma) * d_prev;
            d = norm(d_vec)^2;
    
            numeratore = max(0, loss - (f_ref - delta));
            epsilon = 1e-8;
            alpha = beta * (numeratore) / (d + epsilon); %epsilon permette di evitare divisione per 0
            alpha = max(alpha, 1e-5);
            alpha = min(alpha, 5e-1);
            if ~isfinite(alpha)
                alpha = 1e-3;
            end
        
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
            alpha_prev = alpha;
            fprintf('Loss: %f | Alpha: %.8f | Gamma: %.8f | f_best: %.8f | r:%f | R:%f\n', mse_val(epoch,fold), alpha, gamma, f_best, r, R);
    
            % Calcola gli indici di slicing
            idx1 = 1 : numHidden1*N;                   
            idx2 = idx1(end)+1 : idx1(end)+numHidden1; 
            % idx3 = idx2(end)+1 : idx2(end)+numHidden2*numHidden1; 
            % idx4 = idx3(end)+1 : idx3(end)+numHidden2; 
            idx5 = idx2(end)+1 : idx2(end)+M*numHidden1;         
            idx6 = idx5(end)+1 : idx5(end)+M;
    
            % Estrai le porzioni da d_vec
            d_W_h1   = reshape(d_vec(idx1), numHidden1, N);
            d_b_h1   = reshape(d_vec(idx2), numHidden1, 1);
            % d_W_h2   = reshape(d_vec(idx3), numHidden2, numHidden1);
            % d_b_h2   = reshape(d_vec(idx4), numHidden2, 1);
            d_W_out  = reshape(d_vec(idx5), M, numHidden1);
            d_b_out  = reshape(d_vec(idx6), M, 1);
    
            % Input -> Hidden1
            for j = 1:numHidden1
                hidden_layer1(j).bias_weight = hidden_layer1(j).bias_weight + alpha * d_b_h1(j);
                for i = 1:N
                    hidden_layer1(j).input_connections(i).weight = ...
                        hidden_layer1(j).input_connections(i).weight + ...
                            alpha * (d_W_h1(j,i) - lambda * sign(hidden_layer1(j).input_connections(i).weight));
                end
            end
        
            % Hidden1 -> Hidden2
            % for j = 1:numHidden2
            %     hidden_layer2(j).bias_weight = hidden_layer2(j).bias_weight + alpha * d_b_h2(j);
            %     for i = 1:numHidden1
            %         hidden_layer2(j).input_connections(i).weight = ...
            %             hidden_layer2(j).input_connections(i).weight  ...
            %                 + alpha *  (d_W_h2(j,i) -lambda * sign(hidden_layer2(j).input_connections(i).weight));
            %     end
            % end
        
            % Hidden1 -> Output
            for k = 1:M
                output_layer(k).bias_weight = output_layer(k).bias_weight + alpha * d_b_out(k);
                for j = 1:numHidden1
                    output_layer(k).input_connections(j).weight = ...
                        output_layer(k).input_connections(j).weight + ...
                            alpha * (d_W_out(k,j) - lambda * sign(output_layer(k).input_connections(j).weight));
                end
            end
         
            % Compute Mean Euclidian Error over an epoch
            fprintf('Epoch %d | Vaidation accuracy = %.6f | Train accuracy = %.6f | best_mse : %.6f\n', epoch, acc_val(epoch, fold), acc_train(epoch, fold), best_val_acc(fold));
    
            % Early Stopping based on validation accuracy 
            % if acc_val(epoch, fold) > best_val_acc(fold)
            %     best_val_acc(fold) = acc_val(epoch, fold);
            %     epochs_since_improvement = 0;
            % else
            %     epochs_since_improvement = epochs_since_improvement + 1;
            % end
            % 
            if acc_val(epoch,fold) > best_val_acc(fold)
                best_val_acc(fold) = acc_val(epoch,fold);
            
                best_input_layer{fold} = input_layer;
                best_hidden_layer1{fold} = hidden_layer1;
                best_output_layer{fold}  = output_layer;
            end
            if acc_val(epoch,fold) == 1.0
                break;
            end
            d_prev = d_vec;
        end
        input_layer_final{fold}  = input_layer;
        hidden_layer1_final{fold} = hidden_layer1;
        output_layer_final{fold}  = output_layer;
    end

    training_end_time = posixtime(datetime('now'));

    P_test = size(X_oh_test,1);
    M = 1;
    
    % Matrice per salvare predizioni di tutti i fold
    Yhat_folds = zeros(P_test, k);
    
    for fold = 1:k
        input_layer = best_input_layer{fold};
        hidden_layer1 = best_hidden_layer1{fold};
        output_layer = best_output_layer{fold};
        
        Yhat_test = zeros(P_test, M);
        
        for p = 1:P_test
            % Feedforward input
            for i = 1:size(X_oh_test,2)
                input_layer(i).output = X_oh_test(p,i);
            end
            for j = 1:numHidden1
                hidden_layer1(j).compute();
            end
            for k_out = 1:M
                output_layer(k_out).compute();
                Yhat_test(p,k_out) = output_layer(k_out).output;
            end
        end
        
        Yhat_folds(:,fold) = Yhat_test;
    end
    
    Yhat_ensemble = mean(Yhat_folds, 2) > 0.5;
    
    % Accuracy ensemble
    ensemble_acc = mean(Yhat_ensemble == Y_test);
    fprintf('Test accuracy ensemble: %.4f\n', ensemble_acc);

    % Saving model's data
    model.acc_min = min(nanmean(acc_val,2));
    model.acc_train_final = nanmean(acc_train(end,:));
    model.acc_val_final = nanmean(acc_val(end,:));

    model.lambda = lambda;
    model.numHidden1 = numHidden1;

    model.input_layer_initial = input_layer_initial;
    model.hidden_layer1_initial = hidden_layer1_initial;
    model.output_layer_initial = output_layer_initial;

    model.input_layer_final = input_layer_final;
    model.hidden_layer1_final = hidden_layer1_final;
    model.output_layer_final = output_layer_final;

    model.training_time = training_end_time - training_start_time;

    if ~exist('models', 'dir')
        mkdir('models');
    end

    filename = sprintf( ...
        'models/SGPTL/deflectionRestrictedSGPTL_Monk3-h1-%d-h2-%d-lambda-%g-beta-%g-R-%g-rho-%g-deltaMult-%g_%d.mat', ...
        numHidden1, numHidden2, lambda, beta, R, rho, delta_multiplicator, randi(1e6));
    save(filename, 'model');

    % Plotting and saving the learning curve
    mean_train_curve = nanmean(acc_train,2);
    mean_val_curve   = nanmean(acc_val,2);

    % epochs_per_fold = sum(~isnan(acc_train),1);
    % max_common_epoch = min(epochs_per_fold);
    % 
    % mean_train_curve = mean(acc_train(1:max_common_epoch,:), 2, 'omitnan');
    % mean_val_curve   = mean(acc_val(1:max_common_epoch,:),   2, 'omitnan');

    max_trained_epoch = max(sum(~isnan(acc_train),1));

    [~, name, ~] = fileparts(filename);
    plot_file = fullfile('models/SGPTL', [name '_plot.png']);
    
    fig = figure('Visible','off');
    plot(1:max_trained_epoch, mean_train_curve(1:max_trained_epoch), 'b', 'LineWidth', 2); hold on;
    plot(1:max_trained_epoch, mean_val_curve(1:max_trained_epoch), 'r', 'LineWidth', 2);
    yline(ensemble_acc, 'g', 'LineWidth', 2, 'Label', 'Ensemble Test Acc');
    xlabel('Epoch'); ylabel('Accuracy');
    title(sprintf('Learning Curve |  h1 = %d; h2 = %d; lambda = %g; beta: %g; deltaMul: %g; R: %g; rho: %g; Batch', ...
        numHidden1, numHidden2, lambda, beta, delta_multiplicator, R, rho));
    grid on;
    
    exportgraphics(fig, plot_file);
    close(fig);

    score =  mean(best_val_acc); 
%%
    function hidden_conns = generate_hidden_conns_from(input_units, num_neurons)
        % Glorot/Xavier Initialization
        fan_in = numel(input_units);
        fan_out = num_neurons;

        % Deviazione standard Xavier
        xavier_std = sqrt(2 / (fan_in + fan_out));

        hidden_conns(1, fan_in) = struct('neuron',[],'weight',[]);
        for n = 1:fan_in
            hidden_conns(n).neuron = input_units(n);
            hidden_conns(n).weight = randn * xavier_std;
        end
    end
    
    function output_conns = generate_output_conns_from(hidden_units)
        % Kaiming Initialization
        fan_in = numel(hidden_units);

        output_conns(1, fan_in) = struct('neuron',[],'weight',[]);
        for n = 1:fan_in
            output_conns(n).neuron = hidden_units(n);
            output_conns(n).weight = randn * 0.01;
        end
    end
end
