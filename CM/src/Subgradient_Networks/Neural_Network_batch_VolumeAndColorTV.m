function score = Neural_Network_batch_VolumeAndColorTV(numHidden1, numHidden2, activation_function, lambda, initial_beta, cg, cy, cr, tau0, tau_p, tau_f, tau_min, m_ss, patience, tolerance, init_w)
    %% ===================================
    % LOADING TRAINING DATA (500 patterns)
    % ====================================
    Dataset_TR = readtable('../../data/TR/ML-CUP25-TR.csv');
    
    inputs_TR  = Dataset_TR{:, 2:13};
    outputs_TR = Dataset_TR{:, 14:end};
    
    % Network parameters
    % numHidden1                                % # of units inside first Hidden Layer
    % numHidden2                                % # of units inside second Hidden Layer
    % activation_function                       % activation function of layers
    % lambda                                    % factor for L1 Regularization

    % ColorTV rule parameters
    % initial_beta                              % beta initial value
    % cg, cr, cy                                % ColorTV algorithm parameters, used as threshold to regolize beta value
    
    % Volume Algorithm parameters
    % tau0                                      % Volume algorithm parameter, used to initialize tau as threshold for gamma, when gamma > 1
    % tau_p                                     % Volume algorithm parameter, used to define when update tau
    % tau_f                                     % Volume algorithm parameter, used as rate to update tau
    % tau_min                                   % Volume algorithm parameter, used as threshold for tau
    % m_ss                                      % Volume algorithm parameter, used to determine Serious/Null step

    % Early Stopping parameters
    % patience                                  % # of epoch until last loss improvement            
    % tolerance                                 % threshold of improvement
    
    maxEpochs = 10000;
    
    %% ===================================
    % K-FOLD CROSS-VALIDATION
    % ====================================

    k = 5;

    % RMSE measure collections for train, validation and test
    rmse_train = nan(maxEpochs, k);
    rmse_val   = nan(maxEpochs, k);
    rmse_test   = nan(maxEpochs, k);
    best_train_rmse = inf(1,k);
    best_val_rmse = nan(1,k);
    best_test_rmse = nan(1,k);
    
    % Final epoch per fold
    final_epoch = inf(1,k);

    % --- Hold-out splitting of dataset (80% training + validation, 20% test) ---
    cv_outer = cvpartition(size(inputs_TR,1), 'HoldOut', 0.2);

    trainval_idx = training(cv_outer);
    test_idx     = test(cv_outer);

    A_trainval = inputs_TR(trainval_idx,:);
    B_trainval = outputs_TR(trainval_idx,:);

    A_test = inputs_TR(test_idx,:);
    B_test = outputs_TR(test_idx,:);

    % --- K-Fold Cross-Validation (Fold=5) ---
    cv = cvpartition(size(A_trainval,1), 'KFold', k);

    % Weights definition
    best_W1 = cell(1,k); best_b1 = cell(1,k);
    best_W2 = cell(1,k); best_b2 = cell(1,k);
    best_W3 = cell(1,k); best_b3 = cell(1,k);

    init_W1 = cell(1,k); 
    init_W2 = cell(1,k); 
    init_W3 = cell(1,k);

    % TRAINING START MEASURAMENT
    training_start_time = posixtime(datetime('now'));

    for fold = 1:k

        epoch = 0;

        % Early Stopping parameters initialization
        best_train_rmse(fold) = inf;
        final_epoch(fold) = 0;
        epochs_since_improvement = 0;

        %% ===================================
        % NEURAL NETWORK CONFIGURATION (fully connected)
        % ====================================

        % Weights initialization
        W1 = init_w.W1;
        W2 = init_w.W2;
        W3 = init_w.W3;
        b1 = init_w.b1;
        b2 = init_w.b2;
        b3 = init_w.b3;

        init_W1{fold} = W1; 
        init_W2{fold} = W2; 
        init_W3{fold} = W3;

        %% ===================================
        % I/O NORMALIZATION (zero-mean / unit-variance)
        % ====================================
        train_idx = training(cv, fold);
        val_idx   = test(cv, fold);
     
        A_train_raw = A_trainval(train_idx,:);
        A_val_raw   = A_trainval(val_idx,:);

        mu_A  = mean(A_train_raw,1);
        std_A = std(A_train_raw,0,1);
        std_A = max(std_A, 1e-8);

        A_train = (A_train_raw - mu_A) ./ std_A;
        A_validation   = (A_val_raw   - mu_A) ./ std_A;

        P_train = size(A_train,1);

        B_train = B_trainval(train_idx,:);
        B_validation = B_trainval(val_idx,:);

        mu_B  = mean(B_train,1);
        std_B = std(B_train,0,1);
        std_B = max(std_B, 1e-8);

        B_train_norm = (B_train - mu_B) ./ std_B;
        B_val_norm   = (B_validation - mu_B) ./ std_B;

        %% ===================================
        % BACKPROPAGATION TRAINING LOOP
        % ====================================

        % Feedforward
        [Yhat, A1, Z1, A2, Z2] = Forward(A_train, W1, b1, W2, b2, W3, b3, activation_function);

        % Loss MSE
        mse_loss = mean((Yhat - B_train_norm).^2, 'all');

        % RMSE train
        err = B_train_norm - Yhat;
        rmse_train(epoch, fold) = sqrt(mean(err(:).^2));
        
        % Loss with L1 penalization
        L1 = sum(abs(W1), 'all') + sum(abs(W2), 'all') + sum(abs(W3), 'all');
        loss = mse_loss + lambda * L1;

        % Deflection parameters
        num_params = numel(W1)+numel(b1)+numel(W2)+numel(b2)+numel(W3)+numel(b3);
        d_prev = zeros(num_params, 1);
        loss_prev = loss;

        % ColorTV Parameters initialization
        f_lev = loss * 0.90;        % minimal loss expected
        f_rec = loss;               % 
        ng = 0; ny = 0; nr = 0;     % counters for green, yellow or red step
        rho = 1e-6;                 % threshold to define type of step
        beta = initial_beta;        % beta value to compute deflection value

        % Volume Deflection Parameters
        % Stability Point weights
        W1_bar = W1; b1_bar = b1;
        W2_bar = W2; b2_bar = b2;
        W3_bar = W3; b3_bar = b3;
        f_bar = loss;               % loss 
        g_bar = zeros(num_params, 1);
        sigma = 0;      
        eps_d = 0;      
        tau = tau0;                 % threshold for gamma
        iter_since_tau = 0;         % counter to update tau
        gamma = 1;                  % deflection parameter, at first stage only gradient

        while epoch < maxEpochs
            epoch = epoch + 1;
            
            % Normalized starting gradient
            E_out =  2 * (Yhat - B_train_norm) / (P_train * size(B_train_norm, 2));

            %% Gradient computation
            g = GradientComputation(E_out, A_train, A1, Z1, A2, Z2, W1, W2, W3, activation_function, lambda);

            %% Step di ColorTV
            
            [beta, ng, ny, nr, f_lev, f_rec] = ColorTVRule(loss, loss_prev, d_prev, g, rho, cg, ng, cy, ny, cr, nr, f_lev, f_rec, beta);

            %% Stepsize-restricted Rule
            
            [alpha, d_curr] = StepsizeRestricted(eps_d, sigma, alpha_prev, d_prev, g, gamma_prev, tau, beta, f_lev, loss, epoch);
                
            fprintf('Epoch %d | gamma=%.9f | f_lev=%.6f | alpha=%.9f | loss=%.9f\n', epoch, gamma, f_lev, alpha, loss);
           
            %% Weights update

            [W1, W2, W3, b1, b2, b3] = UpdateWeights(W1, W2, W3, b1, b2, b3, alpha, d_curr);

            %% Feedforward

            [Yhat, A1, Z1, A2, Z2] = Forward(A_train, W1, b1, W2, b2, W3, b3, activation_function);
            loss_prev = loss;
    
            % Loss MSE
            mse_loss = mean((Yhat - B_train_norm).^2, 'all');

            % RMSE train
            err = B_train_norm - Yhat;
            rmse_train(epoch, fold) = sqrt(mean(err(:).^2));
            
            % Loss with L1 penalization
            L1 = sum(abs(W1), 'all') + sum(abs(W2), 'all') + sum(abs(W3), 'all');
            loss = mse_loss + lambda * L1;

            %% Volume Algorithm
            
            [W1_bar, W2_bar, W3_bar, b1_bar, b2_bar, b3_bar, f_bar, g_bar, sigma, eps_d, iter_since_tau, tau] = VolumeAlgorithm(W1_bar, W2_bar, W3_bar, b1_bar, b2_bar, b3_bar, W1, W2, W3, b1, b2, b3, f_bar, g_bar, m_ss, loss, g, sigma, eps_d, iter_since_tau, tau, tau_min, tau_f, tau_p, gamma, d_curr);

            %% Validation
            
            % Feedforward
            [Yval, ~, ~, ~, ~] = Forward(A_validation, W1, b1, W2, b2, W3, b3, activation_function);

            % RMSE validation
            err_val = B_val_norm - Yval;
            rmse_val(epoch, fold) = sqrt(mean(err_val(:).^2));

            if epoch == 1
                best_val_rmse(fold) = rmse_val(epoch, fold);
            end
    
            % Early Stopping based on RMSE 
            if rmse_val(epoch, fold) < best_val_rmse(fold) * (1 - tolerance)
                best_val_rmse(fold) = rmse_val(epoch, fold);
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
                fprintf("EARLY STOP at epoch %d  | RMSE (norm) = %.6f \n", epoch, best_val_rmse(fold));
                break;
            end

            %% Test
            
            % Normalization
            B_test_norm   = (B_test - mu_B) ./ std_B;
            A_test_norm = (A_test - mu_A) ./ std_A;

            % Feedforward
            [Ytest, ~, ~, ~, ~] = Forward(A_test_norm, W1, b1, W2, b2, W3, b3, activation_function);

            % RMSE Test
            err_test = B_test_norm - Ytest;
            rmse_test(epoch, fold) = sqrt(mean(err_test(:).^2));
            
            % Best RMSE test save
            if epoch == 1
                best_test_rmse(fold) = rmse_test(epoch, fold);
            elseif rmse_test(epoch, fold) < best_test_rmse(fold)
                best_test_rmse(fold) = rmse_test(epoch, fold);
            end

            if rmse_train(epoch, fold) < best_train_rmse(fold)
                best_train_rmse(fold) = rmse_train(epoch, fold);
            end

            d_prev = d_curr;
            alpha_prev = alpha;
            gamma_prev = gamma;
        end
    end

    % End of training time
    training_end_time = posixtime(datetime('now'));

    %% Saving model's data
    model.rmse_train = mean(best_train_rmse, 'omitnan');
    model.rmse_validation = mean(best_val_rmse, 'omitnan');
    model.rmse_test = mean(best_test_rmse, 'omitnan');

    model.lambda = lambda;
    model.beta = initial_beta;
    model.cg = cg;
    model.cy = cy;
    model.cr = cr;
    model.tau0 = tau0;
    model.tau_p = tau_p;
    model.tau_f = tau_f;
    model.tau_min = tau_min;
    model.m = m_ss;
    model.stepRes = StepRes;
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

    %% Saving and plot the model results
    if ~exist('models', 'dir')
        mkdir('models');
    end

    avg_best_val = mean(best_val_rmse); 

    if avg_best_val < 0.7
        filename = sprintf( ...
                'models/ColorTV_Volume/stepsize/ColorTV-h1-%d-h2-%d-lambda-%g_%d.mat', ...
                numHidden1, numHidden2, lambda, randi(1e6));
        [~, name, ~] = fileparts(filename);
        plot_file = fullfile('models/ColorTV_Volume/stepsize', [name '_plot.png']);

        save(filename, 'model');

        Plot(rmse_train, rmse_val, rmse_test, avg_best_val, plot_file)
    end

    score = mean(best_val_rmse);
end

function [beta, ng, ny, nr, f_lev, f_rec] = ColorTVRule(loss, loss_prev, d_prev, g, rho, cg, ng, cy, ny, cr, nr, f_lev, f_rec, beta)
    
    delta_f = loss_prev - loss;   
    scal = d_prev' * g;

    if scal > rho && delta_f >= rho * max(abs(f_rec), 1)
        ng = ng+1; ny = 0; nr = 0;
    elseif delta_f >= 0
        ny = ny+1; ng = 0; nr = 0;
    else
        nr = nr+1; ng = 0; ny = 0;
    end
            
    if ng >= cg
       beta = min(2, 2 * beta);
       ng = 0;
    elseif ny >= cy
       beta = min(2, 1.1 * beta);
       ny = 0;
    elseif nr >= cr
       beta = max(5e-4, 0.67 * beta);
       nr = 0;
    end
            
    if loss <= 1.05 * f_lev
       f_lev = f_lev - 0.05 * abs(f_lev);
    end
       f_lev = max(f_lev, 0);
            
    if loss < f_rec
       f_rec = loss;
    end
end

function [alpha, d_curr] = StepsizeRestricted(eps_d, sigma, alpha_prev, d_prev, g, gamma_prev, tau, beta, f_lev, loss, epoch)
    % Deflection()
    if epoch > 1
        num_gamma = eps_d - sigma - alpha_prev * (d_prev(:)' * (g - d_prev));
        den_gamma = alpha_prev * norm(g - d_prev)^2 + 1e-9;
        gamma = num_gamma / den_gamma;
       
        if gamma <= 0
            gamma = 1.0;          % gradiente puro, ricomincia
        elseif gamma > 0 && gamma < 1e-8
            gamma = gamma_prev;
        elseif gamma >= 1
            gamma = min(tau, 1.0);
        end
    end

    % ComputeD()
    d_curr = gamma * g + (1 - gamma) * d_prev;
    % Stepsize-restricted rule
    beta_eff = min(beta, gamma);

    % Stepsize()
    denominatore = norm(d_curr)^2 + 1e-9;
    alpha = max(beta_eff * (loss - f_lev) / denominatore, 0);
end