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
    patience = 400; tolerance = 0.01; maxEpochs = 5000;
    
    %% PERFOMANCE PARAMETERS

    % Folds
    k = 5;
    
    % RMSE (Root Mean Square Error)
    rmse_train = nan(maxEpochs,k); rmse_val = nan(maxEpochs,k); rmse_test = nan(maxEpochs,k);
    best_rmse_test = nan(1,k);
    rmse_test_curve_all = nan(maxEpochs,k);
    
    % MEE (Mean Euclidian Error)
    mee_train = nan(maxEpochs,k); mee_val = nan(maxEpochs,k); mee_val_norm = nan(maxEpochs,k); mee_test = nan(maxEpochs,k);
    best_mee_train = nan(1,k); best_mee_val = nan(1,k); best_mee_test = nan(1,k);
    mee_test_curve_all = nan(maxEpochs,k);

    best_epoch = nan(1,k);
    
    model.weights_init = struct([]);
    model.weights_final = struct([]);
    
    training_start_time = posixtime(datetime('now'));
    
    %% ACTIVATION FUNCTION (Leaky ReLU)
    activation_function = 'leakyrelu';
    
    %% K-FOLD CROSS-VALIDATION LOOP
    cv = cvpartition(size(A_rest,1),'KFold',k);

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
        
        best_val = inf;
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
            Ytr_den = Ytr .* stdB + muB;
            mee_train(epoch,fold) = mean(sqrt(sum((B_tr - Ytr_den).^2,2)));
            
            %% VALIDATION ERRORS
            Yv = Forward(A_vl_norm, W1, b1, W2, b2, W3, b3, activation_function);
            rmse_val(epoch,fold) = sqrt(mean((Yv - B_vl_norm).^2,'all'));
            Yv_den = Yv .* stdB + muB;
            mee_val(epoch,fold) = mean(sqrt(sum((B_vl - Yv_den).^2,2)));
            
            diff_norm = B_vl_norm - Yv;
            mee_val_norm(epoch,fold) = mean(sqrt(sum(diff_norm.^2,2)));
            
            %% INTERNAL TEST ERRORS
            A_test_norm = (A_test - muA) ./ stdA;
            B_test_norm = (B_test - muB) ./ stdB;

            Yt = Forward(A_test_norm, W1, b1, W2, b2, W3, b3, activation_function);
            rmse_test(epoch,fold) = sqrt(mean((Yt - B_test_norm).^2,'all'));
            Yt_den = Yt .* stdB + muB;
            mee_test(epoch,fold) = mean(sqrt(sum((B_test - Yt_den).^2,2)));
            
            %% EARLY-STOPPING (has to improve of 1% wrt the best MEE VL in the last patience epochs)
            if mee_val_norm(epoch,fold) < best_val * (1-tolerance)
                best_val = mee_val_norm(epoch,fold);
                best_mee_val(fold) = mee_val(epoch,fold);
                best_mee_train(fold) = mee_train(epoch,fold);
                best_mee_test(fold) = mee_test(epoch,fold);
                best_rmse_test(fold) = rmse_test(epoch,fold);
                best_epoch(fold) = epoch;

                no_improve = 0;
            else
                no_improve = no_improve + 1;
            end
            
            if no_improve >= patience || isnan(mee_val(epoch,fold))
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
        
        rmse_test_curve_all(:,fold) = rmse_test(:,fold);
        mee_test_curve_all(:,fold) = mee_test(:,fold);
    end

    %% SAVE REST OF MODEL'S DATA
    model.rmse_train_curve_mean = mean(rmse_train, 2, 'omitnan');
    model.rmse_val_curve_mean = mean(rmse_val, 2, 'omitnan');
    model.rmse_test_mean = mean(best_rmse_test,'omitnan');
    
    model.mee_train_curve = mee_train;
    model.mee_val_curve = mee_val;
    model.mee_test_curve = mee_test;
    
    model.best_mee_train_per_fold = best_mee_train;
    model.best_mee_val_per_fold = best_mee_val;
    model.best_mee_test_per_fold = best_mee_test;
    
    model.mee_train_mean = mean(best_mee_train,'omitnan');
    model.mee_cv_mean = mean(best_mee_val,'omitnan');
    model.mee_test_mean = mean(best_mee_test,'omitnan');
    
    model.eta = eta;
    model.alpha = alpha;
    model.lambda = lambda;
    model.batch_size = batch_size;
    model.numHidden1 = numHidden1;
    model.numHidden2 = numHidden2;

    avg_best_mee = mean(best_mee_val,'omitnan');
    
    model.training_time = posixtime(datetime('now')) - training_start_time;
    
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
    Plot(rmse_train, rmse_val, rmse_test_curve_all, avg_best_mee, plot_file);
    
    % mean of MEE VL (denormalized) as model evaluation parameter
    score = model.mee_cv_mean;
end