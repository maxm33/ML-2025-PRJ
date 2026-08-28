function [lambda_star, results, QRsolver] = Least_Squares_QR()
%   M2 MODEL SELECTION Ridge regression model selection using 5-fold CV.
%
%   [lambda_star, results, QRsolver] = Least_Squares_QR()
%
%   Outputs:
%       lambda_star - selected lambda minimizing mean validation RMSE.
%       results     - structure containing all RMSE results and lambdas.
%       QRsolver    - function handle to the nested Householder QR function
%
%   The CSV is expected to have:
%       columns 2:13  -> input features X (12 features)
%       columns 14:17 -> targets Y (4 outputs)
%
%   The function performs:
%       1. 80/20 hold-out split
%       2. 5-fold cross-validation on the training/validation set
%       3. Training-fold normalization
%       4. Ridge regression solved using thin QR
%       5. Selection of lambda based on mean validation RMSE
%       6. Generation of the model-selection plot

    function [Q, R] = computeThinQR(A)
        [m, n] = size(A);

        if n == 0
            Q = zeros(m, 0);
            R = [];
            return;
        end

        x = A(:,1);
        s = -sign(x(1)) * norm(x);
        if s == 0
            s = norm(x); % evita collasso dello shift se x(1) == 0
        end
        e1 = zeros(m,1);
        e1(1) = s;
        v = x - e1;

        if norm(v) > 1e-12
            v = v / norm(v);
        else
            v = zeros(m,1);
        end

        A_transf = A - 2 * v * (v' * A);

        [Q_new, R_new] = computeThinQR(A_transf(2:end, 2:end));

        R = [A_transf(1,1), A_transf(1,2:end); zeros(n-1,1), R_new];
        
        Q_sub = [1, zeros(1, n-1); zeros(m-1, 1), Q_new];

        Q = Q_sub - 2 * v * (v' * Q_sub);
    end

    rootDir = fileparts(mfilename('fullpath'));
    data = readmatrix(fullfile(rootDir, '..', '..', 'data', 'TR', 'ML-CUP25-TR.csv'));

    X = data(:, 2:13);
    Y = data(:, 14:17);

    d = size(X, 2);          % numero di feature (12, senza bias)
    n_outputs = size(Y, 2);  % numero di output (4)

    %% Hold-out: 80% training+validation, 20% test
    cv_outer = cvpartition(size(X,1), 'HoldOut', 0.2);
    trainval_idx = training(cv_outer);
    test_idx = test(cv_outer);

    X_trainval = X(trainval_idx, :);
    Y_trainval = Y(trainval_idx, :);
    X_test     = X(test_idx, :);
    Y_test     = Y(test_idx, :);

    %% Griglia di lambda da testare
    lambdas = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000];

    %% 5-fold Cross-Validation
    k = 5;

    cv = cvpartition(size(X_trainval, 1), 'KFold', k);

    rmse_train_folds = nan(length(lambdas), k);
    rmse_val_folds   = nan(length(lambdas), k);

    for i = 1:length(lambdas)

        lambda = lambdas(i);

        for fold = 1:k
            train_idx = training(cv, fold);
            val_idx   = test(cv, fold);

            X_train_raw = X_trainval(train_idx, :);
            X_val_raw   = X_trainval(val_idx, :);
            Y_train_raw = Y_trainval(train_idx, :);
            Y_val_raw   = Y_trainval(val_idx, :);

            % --- Normalizzazione: statistiche calcolate SOLO sul training fold ---
            X_mean = mean(X_train_raw);
            X_std  = std(X_train_raw);
            X_std  = max(X_std, 1e-8);

            Y_mean = mean(Y_train_raw);
            Y_std  = std(Y_train_raw);
            Y_std  = max(Y_std, 1e-8);

            Xn_train = (X_train_raw - X_mean) ./ X_std;
            Xn_val   = (X_val_raw   - X_mean) ./ X_std;

            Yn_train = (Y_train_raw - Y_mean) ./ Y_std;
            Yn_val   = (Y_val_raw   - Y_mean) ./ Y_std;

            % Aggiunta colonna di bias
            Xb_train = [ones(size(Xn_train,1),1), Xn_train];
            Xb_val   = [ones(size(Xn_val,1),1),   Xn_val];

            n_samples = size(Xb_train, 1);  % numero di campioni nel training fold

            %% Ridge regression via augmented matrix + QR
            %
            % Solves:
            %
            % min ||X*theta - Y||^2 + n_samples*lambda*||theta||^2
            %
            % The bias is also regularized here, as in the original code.

            X_aug = [Xb_train; sqrt(n_samples * lambda) * eye(d+1)];
            Y_aug = [Yn_train; zeros(d+1, n_outputs)];

            [Q, R] = computeThinQR(X_aug);
            theta = R \ (Q' * Y_aug);

            % --- RMSE training ---
            Yhat_train = Xb_train * theta;
            rmse_train_folds(i, fold) = sqrt(mean((Yhat_train - Yn_train).^2, 'all'));

            % --- RMSE validation ---
            Yhat_val = Xb_val * theta;
            rmse_val_folds(i, fold) = sqrt(mean((Yhat_val - Yn_val).^2, 'all'));
        end
    end

    % Media e deviazione standard sui fold
    rmse_train_mean = mean(rmse_train_folds, 2);
    rmse_train_std  = std(rmse_train_folds, 0, 2);
    rmse_val_mean = mean(rmse_val_folds, 2);
    rmse_val_std  = std(rmse_val_folds, 0, 2);

    %% Select lambda*
    [~, best_idx] = min(rmse_val_mean);

    lambda_star = lambdas(best_idx);

    %% Print results
    fprintf('\n');
    fprintf('%12s | %12s | %12s | %12s | %12s\n', ...
        'lambda', 'RMSE train', 'std train', ...
        'RMSE val', 'std val');
    fprintf('      %s\n', repmat('-', 1, 66));

    for i = 1:length(lambdas)
        fprintf('%12.4g | %12.5f | %12.5f | %12.5f | %12.5f\n', ...
            lambdas(i), ...
            rmse_train_mean(i), ...
            rmse_train_std(i), ...
            rmse_val_mean(i), ...
            rmse_val_std(i));
    end

    fprintf('\nlambda* selected = %g ', lambda_star);
    fprintf('(RMSE val = %.5f +/- %.5f)\n', rmse_val_mean(best_idx), rmse_val_std(best_idx));

    %% Grafico: training vs validation RMSE al variare di lambda (scala log-log)
    lambdas_plot = lambdas;
    lambdas_plot(lambdas_plot == 0) = 1e-6; % per poter usare scala log (lambda=0 non rappresentabile)

    figure;
    errorbar(log10(lambdas_plot), ...
             rmse_train_mean, ...
             rmse_train_std, ...
             '-o', ...
             'LineWidth', 1.5);
    hold on;
    errorbar(log10(lambdas_plot), ...
             rmse_val_mean, ...
             rmse_val_std, ...
             '-s', ...
             'LineWidth', 1.5);
    xlabel('log_{10}(\lambda)');
    ylabel('RMSE (normalizzato)');
    legend('Training', 'Validation', 'Location', 'best');
    title('Model selection M2: RMSE training vs validation al variare di \lambda');
    grid on;

    exportgraphics(gcf, fullfile(rootDir, 'M2_model_selection.pdf'), 'ContentType', 'vector');

    %% Return results also in a structure
    results = struct();
    results.lambdas = lambdas;
    results.rmse_train_folds = rmse_train_folds;
    results.rmse_val_folds = rmse_val_folds;
    results.rmse_train_mean = rmse_train_mean;
    results.rmse_train_std = rmse_train_std;
    results.rmse_val_mean = rmse_val_mean;
    results.rmse_val_std = rmse_val_std;
    results.best_idx = best_idx;
    results.X_trainval = X_trainval;
    results.Y_trainval = Y_trainval;
    results.X_test = X_test;
    results.Y_test = Y_test;

    %% Expose solver's nested function
    QRsolver = @computeThinQR;
end