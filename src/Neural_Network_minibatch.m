function score = Neural_Network_minibatch(numHidden1, numHidden2, eta, lambda, alpha, batch_size)
    %% LOADING DATA
    Dataset_TR  = readtable('../data/TR/ML-CUP25-TR.csv');
    
    inputs_raw  = Dataset_TR{:,2:13};
    outputs_raw = Dataset_TR{:,14:end};
    
    [Ns, N] = size(inputs_raw);
    M = size(outputs_raw,2);
    
    %% TRAINING SETTINGS
    patience  = 10000;
    tolerance = 0.01;
    maxEpochs = 10000;

    k  = 5;
    cv = cvpartition(Ns,'KFold',k);
    
    rmse_train    = nan(maxEpochs,k);
    rmse_val      = nan(maxEpochs,k);
    mee_val       = nan(maxEpochs,k);
    mee_val_norm  = nan(maxEpochs,k);
    best_mee      = nan(1,k);
    best_mee_norm = nan(1,k);
    best_epoch    = nan(1,k);
    
    training_start_time = posixtime(datetime('now'));
    
    %% ACTIVATION FUNCTION
    leaky  = @(x) max(0.01 * x, x);
    dleaky = @(x) (x > 0) + 0.01 * (x <= 0);
    
    %% CROSS VALIDATION
    for fold = 1:k
    
        idx_tr = training(cv,fold);
        idx_vl = test(cv,fold);
    
        A_tr = inputs_raw(idx_tr,:);
        A_vl = inputs_raw(idx_vl,:);
        B_tr = outputs_raw(idx_tr,:);
        B_vl = outputs_raw(idx_vl,:);
    
        %% NORMALIZATION
        muA = mean(A_tr);  stdA = max(std(A_tr),1e-8);
        muB = mean(B_tr);  stdB = max(std(B_tr),1e-8);
    
        A_tr      = (A_tr - muA) ./ stdA;
        A_vl      = (A_vl - muA) ./ stdA;
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
    
        best_val   = inf;
        no_improve = 0;
    
        %% TRAINING LOOP
        for epoch = 1:maxEpochs
            
            % Shuffling patterns
            perm = randperm(P_tr);
            A = A_tr(perm,:);
            B = B_tr_norm(perm,:);
        
            %% MINI-BATCH LOOP
            for mb = 1:batch_size:P_tr
        
                idx = mb:min(mb+batch_size-1, P_tr);
                A_b = A(idx,:);
                B_b = B(idx,:);
                P_b = size(A_b,1);
        
                %% FEEDFORWARD
                Z1 = W1 * A_b' + b1;   H1 = leaky(Z1);
                Z2 = W2 * H1 + b2;     H2 = leaky(Z2);
                Yb = (W3 * H2 + b3)';
        
                %% BACKPROPAGATION OF SIGNAL ERRORS
                E3 = (Yb - B_b)';
                E2 = (W3' * E3) .* dleaky(Z2);
                E1 = (W2' * E2) .* dleaky(Z1);
        
                %% GRADIENTS COMPUTATION
                dW3 = (E3 * H2') / P_b + lambda * sign(W3);
                dW2 = (E2 * H1') / P_b + lambda * sign(W2);
                dW1 = (E1 * A_b) / P_b + lambda * sign(W1);
        
                db3 = mean(E3,2);
                db2 = mean(E2,2);
                db1 = mean(E1,2);
        
                %% WEIGHTS UPDATE (with Heavy Ball)
                velocity_W3 = alpha * velocity_W3 - eta * dW3;  W3 = W3 + velocity_W3;
                velocity_W2 = alpha * velocity_W2 - eta * dW2;  W2 = W2 + velocity_W2;
                velocity_W1 = alpha * velocity_W1 - eta * dW1;  W1 = W1 + velocity_W1;
        
                velocity_b3 = alpha * velocity_b3 - eta * db3;  b3 = b3 + velocity_b3;
                velocity_b2 = alpha * velocity_b2 - eta * db2;  b2 = b2 + velocity_b2;
                velocity_b1 = alpha * velocity_b1 - eta * db1;  b1 = b1 + velocity_b1;
            end
        
            %% TRAINING ERROR (RMSE)
            Z1 = W1 * A_tr' + b1;   H1 = leaky(Z1);
            Z2 = W2 * H1 + b2;      H2 = leaky(Z2);
            Y  = (W3 * H2 + b3)';
        
            err_tr = Y - B_tr_norm;
            rmse_train(epoch,fold) = sqrt(mean(err_tr(:).^2));
        
            %% VALIDATION ERROR (MEE and RMSE)
            H1v = leaky(W1 * A_vl' + b1);
            H2v = leaky(W2 * H1v + b2);
            Yv  = (W3 * H2v + b3)';
        
            err_vl = Yv - B_vl_norm;
            rmse_val(epoch,fold) = sqrt(mean(err_vl(:).^2));
        
            Yv_den = Yv .* stdB + muB;
            mee_val(epoch,fold) = mean(sqrt(sum((B_vl - Yv_den).^2,2)));

            diff_norm = B_vl_norm - Yv;
            mee_val_norm(epoch,fold) = mean(sqrt(sum(diff_norm.^2,2)));
        
            %% EARLY STOP
            if mee_val_norm(epoch,fold) < best_val * (1 - tolerance)
                best_val = mee_val_norm(epoch,fold);
                best_mee_norm(fold) = mee_val_norm(epoch,fold);
                best_mee(fold) = mee_val(epoch,fold);
                best_epoch(fold) = epoch;
                no_improve = 0;
            else
                no_improve = no_improve + 1;
            end
        
            if no_improve >= patience || isnan(mee_val(epoch,fold))
                break
            end
        end
    end
    
    training_end_time = posixtime(datetime('now'));
    
    %% SAVING MODEL
    model.rmse_train_min = min(nanmean(rmse_train,2));
    model.rmse_val_min   = min(nanmean(rmse_val,2));
    model.best_mee_per_fold = best_mee;
    model.best_mee_norm_per_fold = best_mee_norm;
    model.mee_cv_mean = mean(best_mee,'omitnan');
    model.mee_cv_std  = std(best_mee,'omitnan');
    model.best_epoch_per_fold = best_epoch;

    model.eta = eta;
    model.alpha = alpha;
    model.lambda = lambda;
    model.numHidden1 = numHidden1;
    model.numHidden2 = numHidden2;

    model.training_time = training_end_time - training_start_time;
    
    if ~exist('models','dir'), mkdir('models'); end
    
    filename = sprintf('models/h1-%d-h2-%d-eta-%g-lambda-%g-alpha-%g-batch-%g_%d.mat',...
        numHidden1,numHidden2,eta,lambda,alpha,batch_size,randi(1e6));
    save(filename,'model');
    
    %% SAVING LEARNING CURVE PLOTS
    mean_tr  = nanmean(rmse_train,2);
    mean_vl  = nanmean(rmse_val,2);
    mean_mee = nanmean(mee_val_norm,2);
    
    max_epoch = max(sum(~isnan(rmse_train),1));
    
    fig = figure('Visible','off'); hold on;
    plot(1:max_epoch,mean_tr(1:max_epoch),'b','LineWidth',2);
    plot(1:max_epoch,mean_vl(1:max_epoch),'r','LineWidth',2);
    plot(1:max_epoch,mean_mee(1:max_epoch),'g','LineWidth',2);
    xlabel('Epoch'); ylabel('Error'); ylim([0 1.5]);
    title(sprintf('Learning Curves | h1=%d h2=%d eta=%g lambda=%g alpha=%g batch=%g',...
        numHidden1,numHidden2,eta,lambda,alpha,batch_size));
    legend({'RMSE TR','RMSE VL','MEE VL'},'Location','best');
    grid on;
    
    [~,name] = fileparts(filename);
    exportgraphics(fig, fullfile('models',[name '_plot.png']));
    close(fig);
    
    score = mean(best_mee,'omitnan');
end
