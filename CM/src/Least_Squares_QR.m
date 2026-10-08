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
    
    d = size(X, 2);          % numero di feature (12)
    n_outputs = size(Y, 2);  % numero di target (4)
    n_samples = size(X, 1);

    % --- Normalizzazione sull'intero dataset ---
    X_mean = mean(X);
    X_std  = max(std(X), 1e-8);
    Y_mean = mean(Y);
    Y_std  = max(std(Y), 1e-8);

    Xn = (X - X_mean) ./ X_std;
    Yn = (Y - Y_mean) ./ Y_std;

    % Aggiunta colonna di bias
    Xb = [ones(n_samples, 1), Xn];

    %% Griglia di lambda da testare
    lambdas = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000];
    mse_train = zeros(length(lambdas), 1);

    for i = 1:length(lambdas)
        lambda = lambdas(i);

        %% Ridge regression via augmented matrix + QR
        X_aug = [Xb; sqrt(n_samples * lambda) * eye(d+1)];
        Y_aug = [Yn; zeros(d+1, n_outputs)];
        
        [Q, R] = computeThinQR(X_aug);
        theta = R \ (Q' * Y_aug);

        % --- MSE Training ---
        Yhat = Xb * theta;
        mse_train(i) = mean((Yhat - Yn).^2, 'all'); 
    end

    %% Select lambda* (lambda con MSE minimo)
    [~, best_idx] = min(mse_train);
    lambda_star = lambdas(best_idx);

    %% Stampa Risultati
    fprintf('\n');
    fprintf('%12s | %12s\n', 'lambda', 'MSE train');
    fprintf('%s\n', repmat('-', 1, 29));
    for i = 1:length(lambdas)
        fprintf('%12.4g | %12.5f\n', lambdas(i), mse_train(i));
    end
    fprintf('\nlambda* selezionato = %g\n', lambda_star);

    %% Grafico MSE Training al variare di lambda
    lambdas_plot = lambdas;
    lambdas_plot(lambdas_plot == 0) = 1e-6; % Per scala log con lambda = 0

    figure;
    plot(log10(lambdas_plot), mse_train, '-o', 'LineWidth', 1.5);
    xlabel('log_{10}(\lambda)');
    ylabel('MSE Training (normalizzato)');
    title('Model Selection M2: MSE Training al variare di \lambda');
    grid on;
    
    exportgraphics(gcf, fullfile(rootDir, 'M2_model_selection.pdf'), 'ContentType', 'vector');

    %% Struct di Output
    results = struct();
    results.lambdas = lambdas;
    results.mse_train = mse_train;
    results.best_idx = best_idx;

    QRsolver = @computeThinQR;
end