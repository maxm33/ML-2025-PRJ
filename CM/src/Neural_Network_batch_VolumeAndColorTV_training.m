function score = Neural_Network_batch_VolumeAndColorTV_training(numHidden1, numHidden2, activation_function, lambda, initial_beta, cg, cy, cr, tau0, tau_p, tau_f, tau_min, m_ss, patience, tolerance, seed, init_w, use_deflection, maxEpochs)

    %% MAKE SHARED LIBRARY FUNCTIONS AVAILABLE
    rootDir = fileparts(mfilename('fullpath'));
    libDir = fullfile(rootDir, '..', '..', 'lib');
    if ~contains(path, libDir)
        addpath(genpath(libDir));
    end

    rng(seed, 'twister');

    %% ===================================
    % LOADING TRAINING DATA (500 patterns)
    % ====================================
    Dataset_TR = readtable(fullfile(rootDir, '..', '..', 'data', 'TR', 'ML-CUP25-TR.csv'));
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

    model.weights_init = struct();
    model.weights_final = struct();
    model.weights_best = struct();

    % Loss curve
    loss_history = nan(maxEpochs, 1);
    gamma_history = nan(maxEpochs, 1);

    % TRAINING START MEASURAMENT
    training_start_time = posixtime(datetime('now'));

    % Early Stopping parameters initialization
    best_train_loss = inf;
    final_epoch = 0;
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

    init_W1 = W1; 
    init_W2 = W2; 
    init_W3 = W3;

    % SAVE INITIAL WEIGHTS
    model.weights_init.W1 = W1;
    model.weights_init.W2 = W2;
    model.weights_init.W3 = W3;

    model.weights_init.b1 = b1;
    model.weights_init.b2 = b2;
    model.weights_init.b3 = b3;

    %% ===================================
    % I/O NORMALIZATION (zero-mean / unit-variance)
    % ====================================
     
    % 1. NORMALIZZAZIONE INPUT (A)
    mu_A  = mean(inputs_TR, 1);
    std_A = std(inputs_TR, 0, 1);
    std_A = max(std_A, 1e-8); 
    
    A_train = (inputs_TR - mu_A) ./ std_A;
    P_train = size(A_train, 1);

    % 2. NORMALIZZAZIONE OUTPUT (B)
    mu_B  = mean(outputs_TR, 1);
    std_B = std(outputs_TR, 0, 1);
    std_B = max(std_B, 1e-8); 
    
    B_train_norm = (outputs_TR - mu_B) ./ std_B;

    %% ===================================
    % INITIAL SETUP & DEFLECTION INITIALIZATION
    % ====================================

    % Feedforward
    [Yhat, A1, Z1, A2, Z2] = Forward(A_train, W1, b1, W2, b2, W3, b3, activation_function);

    % Loss MSE
    mse_loss = mean((Yhat - B_train_norm).^2, 'all');
        
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
    gamma_prev = 1;
    alpha_prev = 1;

    epoch = 1;

    %% ===================================
    % BACKPROPAGATION TRAINING LOOP
    % ====================================
    while epoch <= maxEpochs
            
        % Normalized starting gradient
        E_out =  2 * (Yhat - B_train_norm) / (P_train * size(B_train_norm, 2));

        %% Gradient computation
        g = GradientComputation(E_out, A_train, A1, Z1, A2, Z2, W1, W2, W3, activation_function, lambda);

        %% Step di ColorTV
            
        [beta, ng, ny, nr, f_lev, f_rec] = ColorTVRule(loss, loss_prev, d_prev, g, rho, cg, ng, cy, ny, cr, nr, f_lev, f_rec, beta);

        %% Stepsize-restricted Rule
            
        [alpha, d_curr, gamma] = StepsizeRestricted(eps_d, sigma, alpha_prev, d_prev, g, gamma_prev, tau, beta, f_lev, loss, epoch, use_deflection);

           
        %% Weights update

        [W1, W2, W3, b1, b2, b3] = SubgradientUpdateWeights(W1, W2, W3, b1, b2, b3, alpha, d_curr);

        %% Feedforward

        [Yhat, A1, Z1, A2, Z2] = Forward(A_train, W1, b1, W2, b2, W3, b3, activation_function);
        loss_prev = loss;
    
        % Loss MSE
        mse_loss = mean((Yhat - B_train_norm).^2, 'all');
            
        % Loss with L1 penalization
        L1 = sum(abs(W1), 'all') + sum(abs(W2), 'all') + sum(abs(W3), 'all');
        loss = mse_loss + lambda * L1;
        loss_history(epoch) = loss;

        %% Volume Algorithm
            
        [W1_bar, W2_bar, W3_bar, b1_bar, b2_bar, b3_bar, f_bar, g_bar, sigma, eps_d, iter_since_tau, tau] = VolumeAlgorithm(W1_bar, W2_bar, W3_bar, b1_bar, b2_bar, b3_bar, W1, W2, W3, b1, b2, b3, f_bar, g_bar, m_ss, loss, g, sigma, eps_d, iter_since_tau, tau, tau_min, tau_f, tau_p, gamma, d_curr);

        %% Early Stopping based on training loss
        if epoch == 1 || loss < best_train_loss * (1 - tolerance)
            best_train_loss = loss;
            epochs_since_improvement = 0;
        
            best_W1 = W1; best_b1 = b1;
            best_W2 = W2; best_b2 = b2;
            best_W3 = W3; best_b3 = b3;
        
            model.weights_best.W1 = best_W1;
            model.weights_best.W2 = best_W2;
            model.weights_best.W3 = best_W3;
        
            model.weights_best.b1 = best_b1;
            model.weights_best.b2 = best_b2;
            model.weights_best.b3 = best_b3;
        else
            epochs_since_improvement = epochs_since_improvement + 1;
        end

        if epochs_since_improvement >= patience
           final_epoch = epoch;
           break;
        end

        d_prev = d_curr;
        alpha_prev = alpha;
        gamma_prev = gamma;
        gamma_history(epoch) = gamma;

        epoch = epoch + 1;

        %% SAVE FINAL WEIGHTS
        model.weights_final.W1 = W1;
        model.weights_final.W2 = W2;
        model.weights_final.W3 = W3;

        model.weights_final.b1 = b1;
        model.weights_final.b2 = b2;
        model.weights_final.b3 = b3;
        
    end

    if final_epoch == 0
        final_epoch = maxEpochs;  % training until maxEpochs
    end

    % End of training time
    training_end_time = posixtime(datetime('now'));

    %% Saving model's data
    model.loss_history = loss_history(1:final_epoch);
    model.gamma_history = gamma_history(1:final_epoch);

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
    model.numHidden1 = numHidden1;
    model.numHidden2 = numHidden2;
    model.early_stopping.patience = patience;
    model.early_stopping.tolerance = tolerance;
    model.final_epoch = final_epoch;
    model.hidden1_activation = activation_function;
    model.hidden2_activation = activation_function;
    model.output_activation = 'linear';
    model.seed = seed;
    model.best_loss = best_train_loss;

    model.initial_weights.W1 = init_W1;
    model.initial_weights.W2 = init_W2;
    model.initial_weights.W3 = init_W3;

    model.training_time = training_end_time - training_start_time;
    model.time_per_epoch = model.training_time / final_epoch;

    model.mean_oscillation = compute_oscillation(loss_history, final_epoch);

    %% Saving and plot the model results

    if true

        modelsDir = fullfile(rootDir, 'models/ColorTV_Volume');
        if ~exist(modelsDir, 'dir')
            mkdir(modelsDir);
        end

        uuid_str = char(java.util.UUID.randomUUID);
        unique_id = uuid_str(1:8);

        filename = fullfile(modelsDir, sprintf( ...
                'ColorTV-h1-%d-h2-%d-lambda-%g_%s.mat', ...
                numHidden1, numHidden2, lambda, unique_id));

        save(filename, 'model');

        [~, name] = fileparts(filename);
        
        plot_file = fullfile(modelsDir, [name '_plot.png']);
        Plot_train_loss(loss_history, plot_file);
    end

    score = best_train_loss;
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

function [alpha, d_curr, gamma] = StepsizeRestricted(eps_d, sigma, alpha_prev, d_prev, g, gamma_prev, tau, beta, f_lev, loss, epoch, use_deflection)
    % Deflection()
    if ~use_deflection
        gamma = 1; 
    elseif epoch > 1
        num_gamma = eps_d - sigma - alpha_prev * (d_prev(:)' * (g - d_prev));
        den_gamma = alpha_prev * norm(g - d_prev)^2 + 1e-9;
        gamma = num_gamma / den_gamma;
       
        if gamma <= 0
            gamma = 1.0;       
        elseif gamma > 0 && gamma < 1e-8
            gamma = gamma_prev;
        elseif gamma >= 1
            gamma = min(tau, 1.0);
        end
    else
        gamma = 1;
    end

    % ComputeD()
    d_curr = gamma * g + (1 - gamma) * d_prev;
    % Stepsize-restricted rule
    beta_eff = min(beta, gamma);

    % Stepsize()
    denominatore = norm(d_curr)^2 + 1e-9;
    alpha = max(beta_eff * (loss - f_lev) / denominatore, 0);
end

function osc = compute_oscillation(loss, final_ep)

    series = loss(1:final_ep);
    diffs = diff(series);
    % fraction of epochs where loss increase
    osc = sum(diffs > 0) / length(diffs);
end