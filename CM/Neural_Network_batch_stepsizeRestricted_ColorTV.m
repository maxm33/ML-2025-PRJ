function score = Neural_Network_batch_stepsizeRestricted_ColorTV(numHidden1, numHidden2, lambda, alpha_def)
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
    patience = 100;              
    tolerance = 0.001;                           % minimum of 0.1% progress in patience epochs
    maxEpochs = 10000;
    
    %% ===================================
    % K-FOLD CROSS-VALIDATION
    % ====================================

    k = 5;

    rmse_train = nan(maxEpochs, k);
    mee_train    = nan(maxEpochs, k);
    rmse_val   = nan(maxEpochs, k);
    mee_val    = nan(maxEpochs, k);
    rmse_test   = nan(maxEpochs, k);
    mee_test    = nan(maxEpochs, k);
    best_train_mee = inf(1,k);
    best_val_rmse = nan(1,k);
    best_val_mee = inf(1,k);
    best_test_rmse = nan(1,k);
    best_test_mee = inf(1,k);
    final_epoch = inf(1,k);

    training_start_time = posixtime(datetime('now'));

    cv_outer = cvpartition(size(inputs_TR,1), 'HoldOut', 0.2);

    trainval_idx = training(cv_outer);
    test_idx     = test(cv_outer);

    A_trainval = inputs_TR(trainval_idx,:);
    B_trainval = outputs_TR(trainval_idx,:);

    A_test = inputs_TR(test_idx,:);
    B_test = outputs_TR(test_idx,:);

    cv = cvpartition(size(A_trainval,1), 'KFold', k);
    best_W1 = cell(1,k); best_b1 = cell(1,k);
    best_W2 = cell(1,k); best_b2 = cell(1,k);
    best_W3 = cell(1,k); best_b3 = cell(1,k);

    init_W1 = cell(1,k); 
    init_W2 = cell(1,k); 
    init_W3 = cell(1,k); 

    for fold = 1:k

        train_idx = training(cv, fold);
        val_idx   = test(cv, fold);

        B_train = B_trainval(train_idx,:);
        B_validation = B_trainval(val_idx,:);

        best_train_mee(fold) = inf;
        best_val_mee(fold) = inf;
        best_test_mee(fold) = inf;
        final_epoch(fold) = 0;
        epochs_since_improvement = 0;
    
        mee_history = nan(1, maxEpochs);
        rmse_history = nan(1, maxEpochs);

        %% ===================================
        % NEURAL NETWORK CONFIGURATION (fully connected)
        % ====================================

        % Inizializzazione di Xavier (per tahn)
        W1 = randn(numHidden1, N) * sqrt(1/N); 
        W2 = randn(numHidden2, numHidden1) * sqrt(1/numHidden1);

        % Inizializzazione HE (per ReLU)
        % Hidden Layer 1
        %W1 = randn(numHidden1, N) * sqrt(2/N); 
        b1 = zeros(numHidden1, 1);
        
        % Hidden Layer 2
        %W2 = randn(numHidden2, numHidden1) * sqrt(2/numHidden1);
        b2 = zeros(numHidden2, 1);
        
        % Output Layer (Xavier)
        W3 = randn(M, numHidden2) * sqrt(2/(numHidden2 + M));
        b3 = zeros(M, 1);

        init_W1{fold} = W1; 
        init_W2{fold} = W2; 
        init_W3{fold} = W3;

        %% ===================================
        % I/O NORMALIZATION (zero-mean / unit-variance)
        % ====================================
        A_train_raw = A_trainval(train_idx,:);
        A_val_raw   = A_trainval(val_idx,:);

        mu_A  = mean(A_train_raw,1);
        std_A = std(A_train_raw,0,1);
        std_A = max(std_A, 1e-8);

        A_train = (A_train_raw - mu_A) ./ std_A;
        A_validation   = (A_val_raw   - mu_A) ./ std_A;

        P_train = size(A_train,1);

        mu_B  = mean(B_train,1);
        std_B = std(B_train,0,1);
        std_B = max(std_B, 1e-8);

        B_train_norm = (B_train - mu_B) ./ std_B;
        B_val_norm   = (B_validation - mu_B) ./ std_B;

        %% ===================================
        % BACKPROPAGATION TRAINING LOOP
        % ====================================

        epoch = 0;

        % --- Forward pass ---
        Z1 = A_train * W1' + b1'; 
        A1 = tanh(Z1);
        %A1 = max(0.01 * Z1, Z1); 
            
        Z2 = A1 * W2' + b2';
        A2 = tanh(Z2);
        %A2 = max(0.01 * Z2, Z2);
            
        Yhat = A2 * W3' + b3';

        mse_loss = mean(sum((Yhat - B_train_norm).^2, 2)) / M;

        L1 = sum(abs(W1), 'all') + sum(abs(W2), 'all') + sum(abs(W3), 'all');
        num_params = numel(W1) + numel(b1) + numel(W2) + numel(b2) + numel(W3) + numel(b3);

        loss = mse_loss + (lambda / num_params) * L1;

        target = loss * 0.5;
        beta = 1.0;
        f_rec = loss;
        count_red = 0;
        d_prev = zeros(num_params, 1);

        while epoch < maxEpochs
            epoch = epoch + 1;
            
            % --- Calcolo Errore e MEE Training (Vettorializzato) ---
            Yhat_denorm = Yhat .* std_B + mu_B;
            denorm_errors = B_train - Yhat_denorm;
            % Calcola la norma euclidea per ogni riga (pattern) e poi la media
            total_error = sum(sqrt(sum(denorm_errors.^2, 2)));
            mee_train(epoch, fold) = mean(sqrt(sum(denorm_errors.^2, 2))); 
            
            % E_out per Backprop (sui dati normalizzati)
            E_out = (Yhat - B_train_norm);

            % Gradiente Layer 3
            % grad_W3: [M x numHidden2], grad_b3: [M x 1]
            grad_W3 = (E_out' * A2) / P_train;
            grad_b3 = mean(E_out, 1)';
            
            % Delta Layer 2: [P x numHidden2]
            % dA2 = ones(size(Z2));
            % dA2(Z2 < 0) = 0.01;
            % E_h2 = (E_out * W3) .* dA2;
            E_h2 = (E_out * W3) .* (1 - A2.^2);
            
            grad_W2 = (E_h2' * A1) / P_train;
            grad_b2 = mean(E_h2, 1)';
            
            % Delta Layer 1: [P x numHidden1]
            % dA1 = ones(size(Z1));
            % dA1(Z1 < 0) = 0.01;
            % E_h1 = (E_h2 * W2) .* dA1;
            E_h1 = (E_h2 * W2) .* (1 - A1.^2);
            
            grad_W1 = (E_h1' * A_train) / P_train;
            grad_b1 = mean(E_h1, 1)';

            g = [
                grad_W1(:) + (lambda/num_params) * sign(W1(:));
                grad_b1(:);
                grad_W2(:) + (lambda/num_params) * sign(W2(:));
                grad_b2(:);
                grad_W3(:) + (lambda/num_params) * sign(W3(:));
                grad_b3(:)
            ];
    
            d_curr = alpha_def * g + (1 - alpha_def) * d_prev;
            scal = d_prev' * g;

            if loss < f_rec %verde
                f_rec = loss;
                count_red = 0; 
                beta = min(beta * 1.05, 2.0); 
                target = f_rec * 0.5;
            else
                if scal < 0
                    % rosso
                    count_red = count_red + 1;
                    if count_red >= 2 
                        beta = beta * 0.5;
                        count_red = 0;
                    end
                end
            end

            denominatore = norm(d_curr)^2 + 1e-9;
            alpha = beta * (loss - target) / denominatore;

            alpha = min(alpha, 0.05);

            curr = 0;
            
            % W1
            nElem = numel(W1);
            W1 = W1 - alpha * reshape(d_curr(curr + (1:nElem)), size(W1));
            curr = curr + nElem;
            
            % b1
            nElem = numel(b1);
            b1 = b1 - alpha * reshape(d_curr(curr + (1:nElem)), size(b1));
            curr = curr + nElem;
            
            % W2
            nElem = numel(W2);
            W2 = W2 - alpha * reshape(d_curr(curr + (1:nElem)), size(W2));
            curr = curr + nElem;
            
            % b2
            nElem = numel(b2);
            b2 = b2 - alpha * reshape(d_curr(curr + (1:nElem)), size(b2));
            curr = curr + nElem;
            
            % W3
            nElem = numel(W3);
            W3 = W3 - alpha * reshape(d_curr(curr + (1:nElem)), size(W3));
            curr = curr + nElem;
            
            % b3
            nElem = numel(b3);
            b3 = b3 - alpha * reshape(d_curr(curr + (1:nElem)), size(b3));

            %% Feedworward

            % --- Forward pass ---
            Z1 = A_train * W1' + b1'; 
            A1 = tanh(Z1);
            %A1 = max(0.01 * Z1, Z1); 
                
            Z2 = A1 * W2' + b2';
            A2 = tanh(Z2);
            %A2 = max(0.01 * Z2, Z2);
            
            Yhat = A2 * W3' + b3';

            mse_loss = mean(sum((Yhat - B_train_norm).^2, 2)) / M;

            L1 = sum(abs(W1), 'all') + sum(abs(W2), 'all') + sum(abs(W3), 'all');
            num_params = numel(W1) + numel(b1) + numel(W2) + numel(b2) + numel(W3) + numel(b3);

            loss_new = mse_loss + (lambda / num_params) * L1;
            loss = loss_new;

            %fprintf('Loss: %f | Alpha: %.8f \n', loss_new, alpha);

            %% Validation
            Z1_v = A_validation * W1' + b1'; 
            A1_v = tanh(Z1_v); % Cambia da max(0.01*Z1_v, Z1_v)
            Z2_v = A1_v * W2' + b2';
            A2_v = tanh(Z2_v); % Cambia da max(0.01*Z2_v, Z2_v)

            %Z1_v = A_validation * W1' + b1'; 
            %A1_v = max(0.01*Z1_v, Z1_v);
            %Z2_v = A1_v * W2' + b2';        
            %A2_v = max(0.01*Z2_v, Z2_v);
            Yval = A2_v * W3' + b3';

            err_val = B_val_norm - Yval;
            rmse_val(epoch, fold) = sqrt(mean(err_val(:).^2));
            if epoch == 1
                best_val_rmse(fold) = rmse_val(epoch, fold);
            elseif rmse_val(epoch, fold) < best_val_rmse(fold)
                best_val_rmse(fold) = rmse_val(epoch, fold);
            end

            Yval_denorm = Yval .* std_B + mu_B;
            diff_val = B_validation - Yval_denorm;
            mee_val(epoch, fold) = mean(sqrt(sum(diff_val.^2,2))); 

            % RMSE with collected network outputs over an epoch
            err = B_train_norm - Yhat;
            rmse_per_output = sqrt(mean(err.^2, 1));
            rmse_history(epoch) = mean(rmse_per_output);
            rmse_train(epoch, fold) = rmse_history(epoch);
        
            % Compute Mean Euclidian Error over an epoch
            mee_history(epoch) = total_error / P_train;
            %fprintf('Epoch %d | RMSE (norm) = %.6f | MEE (og scale) = %.6f\n', epoch, rmse_history(epoch), mee_history(epoch));
            %disp(rmse_per_output);
    
            % Early Stopping based on MEE (og scale)
            if mee_val(epoch, fold) < best_val_mee(fold) * (1 - tolerance)
                best_val_mee(fold) = mee_val(epoch, fold);
                epochs_since_improvement = 0;
                % Salva le matrici correnti
                best_W1{fold} = W1; best_b1{fold} = b1;
                best_W2{fold} = W2; best_b2{fold} = b2;
                best_W3{fold} = W3; best_b3{fold} = b3;
            else
                epochs_since_improvement = epochs_since_improvement + 1;
            end

            if epochs_since_improvement >= patience
                final_epoch(fold)= epoch;
                fprintf("EARLY STOP at epoch %d  | RMSE (norm) = %.6f | Best MEE = %.6f\n", epoch, best_val_rmse(fold), best_val_mee(fold));
                break;
            end

            B_test_norm   = (B_test - mu_B) ./ std_B;
            A_test_norm = (A_test - mu_A) ./ std_A;

            Z1_t = A_test_norm * W1' + b1'; 
            A1_t = tanh(Z1_t); % Cambia qui
            Z2_t = A1_t * W2' + b2';        
            A2_t = tanh(Z2_t); % Cambia qui

            %Z1_t = A_test_norm * W1' + b1'; 
            %A1_t = max(0.01*Z1_t, Z1_t);
            %Z2_t = A1_t * W2' + b2';        
            %A2_t = max(0.01*Z2_t, Z2_t);
            Ytest = A2_t * W3' + b3';
            err_test = B_test_norm - Ytest;
            rmse_test(epoch, fold) = sqrt(mean(err_test(:).^2));
            
            Ytest_denorm = Ytest .* std_B + mu_B;
            
            % Error test
            err_test_denorm = B_test - Ytest_denorm;
            mee_test(epoch, fold)  = mean(sqrt(sum(err_test_denorm.^2, 2)));
            if epoch == 1
                best_test_rmse(fold) = rmse_test(epoch, fold);
            elseif rmse_test(epoch, fold) < best_test_rmse(fold)
                best_test_rmse(fold) = rmse_test(epoch, fold);
            end

            if mee_test(epoch, fold) < best_test_mee(fold)
                best_test_mee(fold) = mee_test(epoch, fold);
            end

            if mee_train(epoch, fold) < best_train_mee(fold)
                best_train_mee(fold) = mee_train(epoch, fold);
            end

            d_prev = d_curr;
        end
    end

    training_end_time = posixtime(datetime('now'));

    % Saving model's data
    model.rmse_train = min(nanmean(rmse_train,2));
    model.mee_train = best_train_mee;
    model.mee_validation = best_val_mee;
    model.rmse_validation = mean(best_val_rmse, 'omitnan');
    model.rmse_test = mean(best_test_rmse, 'omitnan');
    model.mee_test = best_test_mee;

    model.lambda = lambda;
    model.alpha = alpha_def;
    model.numHidden1 = numHidden1;
    model.numHidden2 = numHidden2;
    model.k = k;
    model.early_stopping.patience = patience;
    model.early_stopping.tolerance = tolerance;
    model.final_epoch = final_epoch;
    model.hidden1_activation = 'tahn';
    model.hidden2_activation = 'tahn';
    model.output_activation = 'linear';

    model.initial_weights.W1 = init_W1;
    model.initial_weights.W2 = init_W2;
    model.initial_weights.W3 = init_W3;

    model.training_time = training_end_time - training_start_time;

    if ~exist('models', 'dir')
        mkdir('models');
    end

    avg_best_val = mean(best_val_rmse); 

    % Procedi al salvataggio e al plot solo se la condizione è soddisfatta
    if avg_best_val < 0.7
        filename = sprintf( ...
            'models/ColorTV/stepsizeRestricted_ColorTV-h1-%d-h2-%d-lambda-%g_%d.mat', ...
            numHidden1, numHidden2, lambda, randi(1e6));
        
        save(filename, 'model');

        % Calcolo curve medie
        epochs_per_fold = sum(~isnan(rmse_train),1);
        max_common_epoch = min(epochs_per_fold);
        mean_train_curve = mean(rmse_train(1:max_common_epoch,:), 2, 'omitnan');
        mean_val_curve   = mean(rmse_val(1:max_common_epoch,:),   2, 'omitnan');
        mean_test_curve  = mean(rmse_test(1:max_common_epoch,:), 2, 'omitnan');

        [~, name, ~] = fileparts(filename);
        plot_file = fullfile('models/ColorTV', [name '_plot.png']);
        
        % Plotting
        fig = figure('Visible','off');
        h1 = plot(1:max_common_epoch, mean_train_curve, 'b', 'LineWidth', 2); hold on;
        h2 = plot(1:max_common_epoch, mean_val_curve, 'r', 'LineWidth', 2);
        h3 = plot(1:max_common_epoch, mean_test_curve, 'g', 'LineWidth', 2); 
        
        xlabel('Epoch'); ylabel('RMSE');
        title(sprintf('Learning Curve | h1=%d; h2=%d; lambda=%g; aplha=%g; valRMSE=%.3f', ...
            numHidden1, numHidden2, lambda, alpha_def, avg_best_val));
        grid on;
        legend([h1 h2 h3], {'Train', 'Validation', 'Test'}, 'Location', 'best', 'FontSize',18);
        
        exportgraphics(fig, plot_file);
        close(fig);
    end

    score = mean(best_val_mee);
end