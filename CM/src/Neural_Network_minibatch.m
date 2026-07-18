function score = Neural_Network_minibatch(numHidden1, numHidden2, eta, lambda, alpha, batch_size)

    %% MAKE SHARED LIBRARY FUNCTIONS AVAILABLE
    rootDir = fileparts(mfilename('fullpath'));
    libDir = fullfile(rootDir, '..', '..', 'lib');
    if ~contains(path, libDir)
        addpath(genpath(libDir));
    end

    %% LOADING TRAINING DATA
    Dataset = readtable(fullfile(rootDir, '..', '..', 'data', 'TR', 'ML-CUP25-TR.csv'));
    inputs_raw = Dataset{:,2:13};
    outputs_raw = Dataset{:,14:end};
    [Ns, N] = size(inputs_raw);
    M = size(outputs_raw,2);
    
    %% DEFINING INTERNAL TEST SET (20% hold-out)
    [A_test, B_test, A_rest, B_rest] = SplitDatasets(inputs_raw, outputs_raw, Ns, 0.2);

    %% EARLY-STOPPING SETTINGS
    patience = 500; tolerance = 0.02; maxEpochs = 10000;
    
    %% PERFOMANCE PARAMETERS

    % Folds
    k = 5;
    
    % Normalized RMSE (Root Mean Square Error)
    rmse_train = nan(maxEpochs,k); rmse_val = nan(maxEpochs,k); rmse_test = nan(maxEpochs,k);
    best_rmse_train = inf(1,k); best_rmse_val = nan(1,k); best_rmse_test = nan(1,k);

    best_epoch = nan(1,k);
    
    model.weights_init = struct([]);
    model.weights_final = struct([]);
    
    %% ACTIVATION FUNCTION (Leaky ReLU)
    activation_function = 'leakyrelu';
    
    %% K-FOLD CROSS-VALIDATION LOOP
    cv = cvpartition(size(A_rest,1),'KFold',k);

    training_start_time = posixtime(datetime('now'));

    for fold = 1:k
        idx_tr = training(cv,fold);
        idx_vl = test(cv,fold);
        
        % DATASETS
        A_tr = A_rest(idx_tr,:);
        B_tr = B_rest(idx_tr,:);
        A_vl = A_rest(idx_vl,:);
        B_vl = B_rest(idx_vl,:);
        
        % NORMALIZATION
        [A_tr_norm, A_vl_norm, B_tr_norm, B_vl_norm, muA, stdA, muB, stdB] = NormalizeDatasets(A_tr, B_tr, A_vl, B_vl);

        % Save normalization parameters for replicability
        model.norm(fold).muA = muA;
        model.norm(fold).stdA = stdA;
        model.norm(fold).muB = muB;
        model.norm(fold).stdB = stdB;
        
        P_tr = size(A_tr,1);
        
        % HE-KAIMING WEIGHTS INITIALIZATION
        [W1, W2, W3, b1, b2, b3, vel_W1, vel_W2, vel_W3, vel_b1, vel_b2, vel_b3] = GradientInitializeWeights(numHidden1, numHidden2, N, M);
        
        % SAVE INITIAL WEIGHTS
        model.weights_init(fold).W1 = W1;
        model.weights_init(fold).W2 = W2;
        model.weights_init(fold).W3 = W3;

        model.weights_init(fold).b1 = b1;
        model.weights_init(fold).b2 = b2;
        model.weights_init(fold).b3 = b3;
        
        no_improve = 0;
        
        % TRAINING LOOP
        for epoch = 1:maxEpochs
            % Shuffling training patterns
            perm = randperm(P_tr);
            A = A_tr_norm(perm,:); 
            B = B_tr_norm(perm,:);
            
            % MINI-BATCH LOOP
            for mb = 1:batch_size:P_tr
                idx = mb:min(mb+batch_size-1,P_tr);
                A_b = A(idx,:);
                B_b = B(idx,:);

            [W1, W2, W3, b1, b2, b3, vel_W1, vel_W2, vel_W3, vel_b1, vel_b2, vel_b3] = GradientUpdateWeights(W1, W2, W3, b1, b2, b3, ...
                    vel_W1, vel_W2, vel_W3, vel_b1, vel_b2, vel_b3, ...
                    A_b, B_b, eta, lambda, alpha, activation_function);
            end

            %% TRAINING ERRORS
            Ytr = Forward(A_tr_norm, W1, b1, W2, b2, W3, b3, activation_function);
            rmse_train(epoch,fold) = sqrt(mean((Ytr - B_tr_norm).^2,'all'));
            
            %% VALIDATION ERRORS
            Yv = Forward(A_vl_norm, W1, b1, W2, b2, W3, b3, activation_function);
            rmse_val(epoch,fold) = sqrt(mean((Yv - B_vl_norm).^2,'all'));
            
            %% INTERNAL TEST ERRORS
            A_test_norm = (A_test - muA) ./ stdA;
            B_test_norm = (B_test - muB) ./ stdB;

            Yt = Forward(A_test_norm, W1, b1, W2, b2, W3, b3, activation_function);
            rmse_test(epoch,fold) = sqrt(mean((Yt - B_test_norm).^2,'all'));
            
            if epoch == 1
                best_rmse_val(fold) = rmse_val(epoch,fold);
                best_rmse_test(fold) = rmse_test(epoch,fold);
            end

            %% EARLY-STOPPING (has to improve of tolerance% wrt the best RMSE VL in the last patience epochs)
            if rmse_val(epoch,fold) < best_rmse_val(fold) * (1-tolerance)
                best_rmse_val(fold) = rmse_val(epoch,fold);
                best_epoch(fold) = epoch;
                no_improve = 0;
            else
                no_improve = no_improve + 1;
            end

            if rmse_train(epoch,fold) < best_rmse_train(fold)
                best_rmse_train(fold) = rmse_train(epoch,fold);
            end

            if rmse_test(epoch,fold) < best_rmse_test(fold)
                best_rmse_test(fold) = rmse_test(epoch,fold);
            end
            
            if no_improve >= patience || isnan(rmse_val(epoch,fold))
                break
            end
        end

        %% SAVE FINAL WEIGHTS
        model.weights_final(fold).W1 = W1;
        model.weights_final(fold).W2 = W2;
        model.weights_final(fold).W3 = W3;

        model.weights_final(fold).b1 = b1;
        model.weights_final(fold).b2 = b2;
        model.weights_final(fold).b3 = b3;
        
    end
    
    %% SAVE REST OF MODEL'S DATA
    model.training_time = posixtime(datetime('now')) - training_start_time;

    model.rmse_train = mean(best_rmse_train, 'omitnan');
    model.rmse_val = mean(best_rmse_val, 'omitnan');
    model.rmse_test = mean(best_rmse_test, 'omitnan');
    
    model.rmse_train_curve = rmse_train;
    model.rmse_val_curve = rmse_val;
    model.rmse_test_curve = rmse_test;
    
    model.eta = eta;
    model.alpha = alpha;
    model.lambda = lambda;
    model.batch_size = batch_size;
    model.numHidden1 = numHidden1;
    model.numHidden2 = numHidden2;

    avg_best_val = mean(best_rmse_val, 'omitnan');

    %% CHECK WHETHER MODEL SHOULD BE SAVED
    VAR_THRESHOLD       = 0.001;     % normalized total variation threshold
    OVERFIT_THRESHOLD   = 0.025;     % allowed validation-training RMSE gap
    RMSE_THRESHOLD      = 0.65;
    
    totalVariations = zeros(1,k);
    overfitGaps = zeros(1,k);
    save_model = true;
    
    for fold = 1:k
        curve = rmse_val(:,fold);
        curve = curve(~isnan(curve));
    
        if numel(curve) > 1
            totalVariations(fold) = sum(abs(diff(curve))) / numel(curve);
        end
    
        overfitGaps(fold) = best_rmse_val(fold) - best_rmse_train(fold);
    end
    
    avgTotalVariation = mean(totalVariations);
    avgOverfitGap = mean(overfitGaps);
    
    if avgTotalVariation > VAR_THRESHOLD ||...
       avgOverfitGap > OVERFIT_THRESHOLD ||...
       avg_best_val > RMSE_THRESHOLD
            
        save_model = false;
    end
    
    if save_model
        modelsDir = fullfile(rootDir, 'models');
        if ~exist(modelsDir, 'dir')
            mkdir(modelsDir);
        end
    
        filename = fullfile(modelsDir, sprintf( ...
            'h1-%d-h2-%d-eta-%g-lambda-%g-alpha-%g-batch-%g_%d.mat', ...
            numHidden1, numHidden2, eta, lambda, alpha, batch_size, randi(1e6)));
    
        save(filename, 'model');
    
        [~, name] = fileparts(filename);
    
        %% PLOT AND SAVE LEARNING CURVES
        plot_file = fullfile(modelsDir, [name '_plot.png']);
        Plot(rmse_train, rmse_val, rmse_test, avg_best_val, plot_file);
    end
    
    % mean of RMSE VALIDATION as model evaluation parameter
    score = avg_best_val;
end