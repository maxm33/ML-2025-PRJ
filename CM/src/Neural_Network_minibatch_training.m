function score = Neural_Network_minibatch_training(numHidden1, numHidden2, activation_function, lambda, initial_eta, cg, cy, cr, alpha, batch_size, seed, patience, tolerance, init_w, maxEpochs)

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

    % Weights definition
    best_W1 = []; best_b1 = [];
    best_W2 = []; best_b2 = [];
    best_W3 = []; best_b3 = [];

    % Loss curve
    loss_history = nan(maxEpochs, 1);

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

    % SAVE INITIAL WEIGHTS
    model.weights_init.W1 = W1;
    model.weights_init.W2 = W2;
    model.weights_init.W3 = W3;
    model.weights_init.b1 = b1;
    model.weights_init.b2 = b2;
    model.weights_init.b3 = b3;

    vel_W1 = zeros(size(W1));
    vel_W2 = zeros(size(W2));
    vel_W3 = zeros(size(W3));
    vel_b1 = zeros(size(b1));
    vel_b2 = zeros(size(b2));
    vel_b3 = zeros(size(b3));

    %% ===================================
    % I/O NORMALIZATION (zero-mean / unit-variance)
    % ====================================
     
    % 1. NORMALIZZAZIONE INPUT (A)
    mu_A  = mean(inputs_TR, 1);
    std_A = std(inputs_TR, 0, 1);
    std_A = max(std_A, 1e-8); 
    
    A_train = (inputs_TR - mu_A) ./ std_A;
    P_tr = size(inputs_TR, 1);

    % 2. NORMALIZZAZIONE OUTPUT (B)
    mu_B  = mean(outputs_TR, 1);
    std_B = std(outputs_TR, 0, 1);
    std_B = max(std_B, 1e-8); 
    
    B_train_norm = (outputs_TR - mu_B) ./ std_B;

    % Feedforward
    [Yf, A1f, Z1f, A2f, Z2f] = Forward(A_train, W1, b1, W2, b2, W3, b3, activation_function);

    % Loss mse
    mse_loss = mean((Yf - B_train_norm).^2, 'all');
    L1 = sum(abs(W1),'all') + sum(abs(W2),'all') + sum(abs(W3),'all');
    loss_curr = mse_loss + lambda * L1;
    
    % ColorTV parameters
    f_lev = loss_curr * 0.99;
    f_rec = loss_curr;
    ng = 0; ny = 0; nr = 0;
    rho = 1e-6;
    eta = initial_eta;
    loss_prev = loss_curr;
    d_prev = zeros(numel(W1)+numel(b1)+numel(W2)+numel(b2)+numel(W3)+numel(b3), 1);

    epoch = 1;

    %% ===================================
    % BACKPROPAGATION TRAINING LOOP
    % ====================================
    while epoch <= maxEpochs

        E_out_full = 2 * (Yf - B_train_norm) / (P_tr * size(B_train_norm, 2));
        g_full = GradientComputation(E_out_full, A_train, A1f, Z1f, A2f, Z2f, ...
            W1, W2, W3, activation_function, lambda);
    
        %% ColorTV rule
        [eta, ng, ny, nr, f_lev, f_rec] = ColorTVRule(loss_curr, loss_prev, ...
            d_prev, g_full, rho, cg, ng, cy, ny, cr, nr, f_lev, f_rec, eta);
    
        denom = norm(g_full)^2 + 1e-9;
        eta_epoch = max(eta * (loss_curr - f_lev) / denom, 0);
            
        % Shuffling training patterns
        perm = randperm(P_tr);
        A = A_train(perm,:); 
        B = B_train_norm(perm,:);
            
        % MINI-BATCH LOOP
        for mb = 1:batch_size:P_tr
            idx = mb:min(mb+batch_size-1,P_tr);
            A_b = A(idx,:);
            B_b = B(idx,:);

            [W1, W2, W3, b1, b2, b3, vel_W1, vel_W2, vel_W3, vel_b1, vel_b2, vel_b3] = GradientUpdateWeights(W1, W2, W3, b1, b2, b3, ...
                 vel_W1, vel_W2, vel_W3, vel_b1, vel_b2, vel_b3, ...
                 A_b, B_b, eta_epoch, lambda, alpha, batch_size, activation_function);
        end

        [Yf, A1f, Z1f, A2f, Z2f] = Forward(A_train, W1, b1, W2, b2, W3, b3, activation_function);
        mse_loss = mean((Yf - B_train_norm).^2, 'all');
        L1 = sum(abs(W1),'all') + sum(abs(W2),'all') + sum(abs(W3),'all');
        loss = mse_loss + lambda * L1;
        loss_history(epoch) = loss;

        %% Early Stopping based on MSE 
        if epoch == 1 || loss < best_train_loss * (1 - tolerance)
            best_train_loss = loss;
            epochs_since_improvement = 0;
            
            best_W1 = W1; best_b1 = b1;
            best_W2 = W2; best_b2 = b2;
            best_W3 = W3; best_b3 = b3;
        else
            epochs_since_improvement = epochs_since_improvement + 1;
        end

        if epochs_since_improvement >= patience
           final_epoch = epoch;
           break;
        end

        epoch = epoch + 1;
        loss_prev = loss_curr;
        loss_curr = loss;
        d_prev = [vel_W1(:); vel_b1(:); vel_W2(:); vel_b2(:); vel_W3(:); vel_b3(:)];
    end

    %% SAVE FINAL WEIGHTS
        model.weights_final.W1 = W1;
        model.weights_final.W2 = W2;
        model.weights_final.W3 = W3;
        model.weights_final.b1 = b1;
        model.weights_final.b2 = b2;
        model.weights_final.b3 = b3;
        
        %% SAVE BEST WEIGHTS
        model.weights_best.W1 = best_W1;
        model.weights_best.W2 = best_W2;
        model.weights_best.W3 = best_W3;
        model.weights_best.b1 = best_b1;
        model.weights_best.b2 = best_b2;
        model.weights_best.b3 = best_b3;

    if final_epoch == 0
        final_epoch = maxEpochs;  % il ciclo è arrivato a maxEpochs
    end

    % End of training time
    training_end_time = posixtime(datetime('now'));

    %% Saving model's data
    model.loss_history = loss_history(1:final_epoch);

    model.eta = initial_eta;
    model.alpha = alpha;
    model.lambda = lambda;
    model.batch_size = batch_size;
    model.maxEpochs = maxEpochs;
    model.numHidden1 = numHidden1;
    model.numHidden2 = numHidden2;
    model.activation = activation_function;
    model.seed = seed;
    model.best_loss = best_train_loss;

    model.training_time = training_end_time - training_start_time;
    model.time_per_epoch = model.training_time / final_epoch;

    model.mean_oscillation = compute_oscillation(loss_history, final_epoch);

    %% Saving and plot the model results

    modelsDir = fullfile(rootDir, 'models/Gradient');
    if ~exist(modelsDir, 'dir')
        mkdir(modelsDir);
    end

    % ID univoco derivato dal thread/worker o UUID (non altera rng)
    uuid_str = char(java.util.UUID.randomUUID);
    unique_id = uuid_str(1:8);

    filename = fullfile(modelsDir, sprintf( ...
            'Gradient-h1-%d-h2-%d-eta-%g-alpha-%g_%s.mat', ...
            numHidden1, numHidden2, eta, alpha, unique_id));

    save(filename, 'model');

    [~, name] = fileparts(filename);
    
    plot_file = fullfile(modelsDir, [name '_plot.png']);
    PlotTrainingLoss(loss_history, plot_file);

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

function osc = compute_oscillation(loss, final_ep)
    series = loss(1:final_ep);
    diffs = diff(series);
    % frazione di epoche in cui la loss aumenta (non-monotonicità)
    osc = sum(diffs > 0) / length(diffs);
end