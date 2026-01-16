function score = Neural_Network_minibatch(numHidden1, numHidden2, eta, lambda, alpha, batch_size)
    %% LOADING DATA
    Dataset = readtable('../data/TR/ML-CUP25-TR.csv');
    
    inputs_raw  = Dataset{:,2:13};
    outputs_raw = Dataset{:,14:end};
    
    [Ns, N] = size(inputs_raw);
    M = size(outputs_raw,2);
    
    %% STOPPING SETTINGS
    patience  = 400;
    tolerance = 0.01;
    maxEpochs = 5000;
    
    %% INTERNAL TEST SET (15% hold-out)
    cv_test = cvpartition(Ns,'HoldOut',0.15);

    idx_test = test(cv_test);
    idx_rest = training(cv_test);
    
    A_test = inputs_raw(idx_test,:);
    B_test = outputs_raw(idx_test,:);
    
    A_rest = inputs_raw(idx_rest,:);
    B_rest = outputs_raw(idx_rest,:);
    
    %% K-FOLD CROSS VALIDATION
    k = 5;
    cv = cvpartition(sum(idx_rest),'KFold',k);
    
    rmse_train = nan(maxEpochs,k);
    rmse_val   = nan(maxEpochs,k);
    rmse_test  = nan(maxEpochs,k);
    
    mee_train    = nan(maxEpochs,k);
    mee_val      = nan(maxEpochs,k);
    mee_val_norm = nan(maxEpochs,k);
    mee_test     = nan(maxEpochs,k);
    
    best_mee_train = nan(1,k);
    best_mee_val   = nan(1,k);
    best_mee_test  = nan(1,k);
    best_rmse_test = nan(1,k);
    best_epoch     = nan(1,k);
    
    model.weights_init  = struct([]);
    model.weights_final = struct([]);

    rmse_test_curve_all = nan(maxEpochs,k);
    mee_test_curve_all  = nan(maxEpochs,k);
    
    training_start_time = posixtime(datetime('now'));
    
    %% ACTIVATION FUNCTION
    leaky  = @(x) max(0.01 * x, x);
    dleaky = @(x) (x > 0) + 0.01 * (x <= 0);
    
    %% CROSS VALIDATION LOOP
    for fold = 1:k
        idx_tr = training(cv,fold);
        idx_vl = test(cv,fold);
        
        %% DATASETS
        A_tr = A_rest(idx_tr,:);
        B_tr = B_rest(idx_tr,:);
        A_vl = A_rest(idx_vl,:);
        B_vl = B_rest(idx_vl,:);
        
        %% NORMALIZATION
        muA = mean(A_tr); stdA = max(std(A_tr),1e-8);
        muB = mean(B_tr); stdB = max(std(B_tr),1e-8);
        
        A_tr_norm = (A_tr - muA) ./ stdA;
        A_vl_norm = (A_vl - muA) ./ stdA;
        B_tr_norm = (B_tr - muB) ./ stdB;
        B_vl_norm = (B_vl - muB) ./ stdB;
        
        P_tr = size(A_tr,1);
        
        %% WEIGHTS INITIALIZATION
        W1 = randn(numHidden1,N) * sqrt(2/N);                   b1 = zeros(numHidden1,1);
        W2 = randn(numHidden2,numHidden1) * sqrt(2/numHidden1); b2 = zeros(numHidden2,1);
        W3 = randn(M,numHidden2) * sqrt(2/numHidden2);          b3 = zeros(M,1);
        
        velocity_W1 = zeros(size(W1)); velocity_b1 = zeros(size(b1));
        velocity_W2 = zeros(size(W2)); velocity_b2 = zeros(size(b2));
        velocity_W3 = zeros(size(W3)); velocity_b3 = zeros(size(b3));
        
        %% SAVE INITIAL WEIGHTS
        model.weights_init(fold).W1 = W1; model.weights_init(fold).b1 = b1;
        model.weights_init(fold).W2 = W2; model.weights_init(fold).b2 = b2;
        model.weights_init(fold).W3 = W3; model.weights_init(fold).b3 = b3;
        
        best_val = inf;
        no_improve = 0;
        
        %% TRAINING LOOP
        for epoch = 1:maxEpochs
            % Shuffling training patterns
            perm = randperm(P_tr);
            A = A_tr_norm(perm,:); 
            B = B_tr_norm(perm,:);
            
            %% MINI-BATCH LOOP
            for mb = 1:batch_size:P_tr
                idx = mb:min(mb+batch_size-1,P_tr);
                A_b = A(idx,:);
                B_b = B(idx,:);
                P_b = size(A_b,1);
                
                Z1 = W1*A_b' + b1;    H1 = leaky(Z1);
                Z2 = W2*H1   + b2;    H2 = leaky(Z2);
                Yb = (W3*H2  + b3)';
                
                % Error signals computation
                E3 = (Yb - B_b)'; E2 = (W3'*E3).*dleaky(Z2); E1 = (W2'*E2).*dleaky(Z1);
                
                % Gradients computation
                dW3 = (E3*H2')/P_b + lambda*sign(W3);
                dW2 = (E2*H1')/P_b + lambda*sign(W2);
                dW1 = (E1*A_b)/P_b + lambda*sign(W1);
                
                db3 = mean(E3,2); db2 = mean(E2,2); db1 = mean(E1,2);
                
                % Momentum velocities + weights update
                velocity_W3 = alpha*velocity_W3 - eta*dW3; W3 = W3 + velocity_W3;
                velocity_W2 = alpha*velocity_W2 - eta*dW2; W2 = W2 + velocity_W2;
                velocity_W1 = alpha*velocity_W1 - eta*dW1; W1 = W1 + velocity_W1;
                
                velocity_b3 = alpha*velocity_b3 - eta*db3; b3 = b3 + velocity_b3;
                velocity_b2 = alpha*velocity_b2 - eta*db2; b2 = b2 + velocity_b2;
                velocity_b1 = alpha*velocity_b1 - eta*db1; b1 = b1 + velocity_b1;
            end
            %% TRAINING ERRORS
            Ytr = (W3*leaky(W2*leaky(W1*A_tr_norm'+b1)+b2)+b3)';
            rmse_train(epoch,fold) = sqrt(mean((Ytr - B_tr_norm).^2,'all'));
            Ytr_den = Ytr .* stdB + muB;
            mee_train(epoch,fold) = mean(sqrt(sum((B_tr - Ytr_den).^2,2)));
            
            %% VALIDATION ERRORS
            Yv = (W3*leaky(W2*leaky(W1*A_vl_norm'+b1)+b2)+b3)';
            rmse_val(epoch,fold) = sqrt(mean((Yv - B_vl_norm).^2,'all'));
            Yv_den = Yv .* stdB + muB;
            mee_val(epoch,fold) = mean(sqrt(sum((B_vl - Yv_den).^2,2)));
            
            diff_norm = B_vl_norm - Yv;
            mee_val_norm(epoch,fold) = mean(sqrt(sum(diff_norm.^2,2)));
            
            %% INTERNAL TEST ERRORS
            A_test_norm = (A_test - muA) ./ stdA;
            B_test_norm = (B_test - muB) ./ stdB;

            Yt = (W3*leaky(W2*leaky(W1*A_test_norm'+b1)+b2)+b3)';
            rmse_test(epoch,fold) = sqrt(mean((Yt - B_test_norm).^2,'all'));
            Yt_den = Yt .* stdB + muB;
            mee_test(epoch,fold) = mean(sqrt(sum((B_test - Yt_den).^2,2)));
            
            %% EARLY STOPPING
            if mee_val_norm(epoch,fold) < best_val*(1-tolerance)
                best_val = mee_val_norm(epoch,fold);
                best_mee_val(fold)   = mee_val(epoch,fold);
                best_mee_train(fold) = mee_train(epoch,fold);
                best_mee_test(fold)  = mee_test(epoch,fold);
                best_rmse_test(fold) = rmse_test(epoch,fold);
                best_epoch(fold)     = epoch;

                no_improve = 0;
            else
                no_improve = no_improve + 1;
            end
            
            if no_improve >= patience || isnan(mee_val(epoch,fold))
                break
            end
        end
        %% SAVE FINAL WEIGHTS
        model.weights_final(fold).W1 = W1; model.weights_final(fold).b1 = b1;
        model.weights_final(fold).W2 = W2; model.weights_final(fold).b2 = b2;
        model.weights_final(fold).W3 = W3; model.weights_final(fold).b3 = b3;
        
        rmse_test_curve_all(:,fold) = rmse_test(:,fold);
        mee_test_curve_all(:,fold)  = mee_test(:,fold);
    end
    %% SAVE MODEL
    model.rmse_train_min = min(nanmean(rmse_train,2));
    model.rmse_val_min   = min(nanmean(rmse_val,2));
    model.rmse_test_mean = mean(best_rmse_test,'omitnan');
    
    model.mee_train_curve = mee_train;
    model.mee_val_curve   = mee_val;
    model.mee_test_curve  = mee_test;
    
    model.best_mee_train_per_fold = best_mee_train;
    model.best_mee_val_per_fold   = best_mee_val;
    model.best_mee_test_per_fold  = best_mee_test;
    
    model.mee_train_mean = mean(best_mee_train,'omitnan');
    model.mee_cv_mean    = mean(best_mee_val,'omitnan');
    model.mee_test_mean  = mean(best_mee_test,'omitnan');
    
    model.eta        = eta;
    model.alpha      = alpha;
    model.lambda     = lambda;
    model.batch_size = batch_size;
    model.numHidden1 = numHidden1;
    model.numHidden2 = numHidden2;
    
    model.training_time = posixtime(datetime('now')) - training_start_time;
    
    if ~exist('models','dir'), mkdir('models'); end
    filename = sprintf('models/h1-%d-h2-%d-eta-%g-lambda-%g-alpha-%g-batch-%g_%d.mat',...
        numHidden1,numHidden2,eta,lambda,alpha,batch_size,randi(1e6));
    save(filename,'model');
    
    %% SAVE LEARNING CURVE PLOTS
    mean_tr = nanmean(rmse_train,2);
    mean_vl = nanmean(rmse_val,2);
    mean_ts = nanmean(rmse_test_curve_all,2);
    
    max_epoch = max(sum(~isnan(rmse_train),1));
    
    fig = figure('Visible','off'); hold on;
    plot(1:max_epoch,mean_tr(1:max_epoch),'b','LineWidth',2);
    plot(1:max_epoch,mean_vl(1:max_epoch),'r','LineWidth',2);
    plot(1:max_epoch,mean_ts(1:max_epoch),'g','LineWidth',2);
    xlabel('Epoch'); ylabel('RMSE');
    legend({'Train','Validation','Test'},'Location','best');
    title(sprintf('Learning Curves | h1=%d h2=%d eta=%g lambda=%g alpha=%g batch=%g',...
        numHidden1,numHidden2,eta,lambda,alpha,batch_size)); grid on;
    
    [~,name] = fileparts(filename);
    exportgraphics(fig, fullfile('models',[name '_plot.png']));
    close(fig);
    
    % mean of MEE VL (OG SCALE) as model evaluation parameter
    score = model.mee_cv_mean;
end
