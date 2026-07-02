data = readmatrix('../../data/TR/ML-CUP25-TR.csv');
dataTS = readmatrix('../../data/TS/ML-CUP25-TS.csv');

X = data(:, 2:13);
Y = data(:, 14:17);

X_ts = dataTS(:, 2:end);

% normalizza
X_mean = mean(X);
X_std = std(X);

Xn = (X - X_mean) ./ X_std;
Xts_n = (X_ts - X_mean) ./ X_std;

Y_mean = mean(Y);
Y_std = std(Y);
Yn = (Y - Y_mean) ./ Y_std;

% inizializzo bias e pesi
X_bias = [ones(size(Xn,1), 1), Xn];

n_outputs = size(Yn, 2);

% utilizzo fattorizzazione QR
lambdas = [0, 0.001, 0.1, 1, 10, 100, 1000, 10000];
for i = 1:length(lambdas)
    lambda = lambdas(i);
    n = size(X_bias, 2);
    X_aug = [X_bias; sqrt(lambda) * eye(n)];
    y_aug = [Yn; zeros(n, n_outputs)];
    [Q, R] = computeThinQR(X_aug);
    x = R \ (Q' * y_aug);
    Y_pred = X_bias * x;
    rmse = sqrt(mean((Y_pred - Yn).^2, 'all'));
    fprintf('lambda = %8.3f | RMSE train = %.5f | norm(x) = %.5f\n', ...
        lambda, rmse, norm(x));
end

figure;
scatter3(Yn(:,1), Yn(:,2), Yn(:,3), 40, 'filled');
title('Shape of Training Targets (Normalized)');
xlabel('Output 1'); ylabel('Output 2'); zlabel('Output 3');
grid on; 

figure;
scatter3(Y_pred(:,1), Y_pred(:,2), Y_pred(:,3), 40, 'filled');
title('Predictions on Training Set (Normalized)');
xlabel('Output 1'); ylabel('Output 2'); zlabel('Output 3');
grid on; 

for k = 1:n_outputs
    figure; hold on;
    scatter(Yn(:,k), Y_pred(:,k), 40, 'blue', 'filled');
    x_vals = linspace(min(Yn(:,k)), max(Yn(:,k)), 200);
    plot(x_vals, x_vals, 'r--', 'LineWidth', 2);
    rmse = sqrt(mean((Y_pred(:,k) - Yn(:,k)).^2));
    xlabel('True Target'); ylabel('Predicted Target');
    title(['Output ' num2str(k) ' - True vs Predicted (Normalized) - RMSE = ' num2str(rmse)]);
    grid on; axis equal;
end


% Yts_pred_norm = Xts_bias * W; 
% size(Xts_bias)
% size(Y_std)
% Yts_pred = Yts_pred_norm .* Y_std + Y_mean

%per calcolare decomposizione QR totale
function [Q, R] = computeQR(A)

    [m, n] = size(A);

    % caso base: n == 0
    if n == 0
        Q = eye(m);
        R = [];
        return;
    end

    % costruzione Householder sulla prima colonna
    x = A(:,1);
    s = -sign(x(1)) * norm(x);
    e1 = zeros(m,1); 
    e1(1) = s;
    v = x - e1;
    if norm(v) > 1e-12 %permette di controllare di non dividere per 0
        v = v / norm(v);
        H = eye(m) - 2*(v*v');
    else
        H = eye(m);
    end

    % applica trasformazione
    A_sub = H*A;

    % parte 2 della ricorsione
    [Q_sub, R_sub] = computeQR(A_sub(2:end, 2:end));

    R = [A_sub(1,1), A_sub(1,2:end);
         zeros(m-1,1), R_sub];
    Q = H * blkdiag(1, Q_sub);
end



%per calcolare decomposizione QR thin
function [Q, R] = computeThinQR(A)
    [m, n] = size(A);
    
    % Caso base: se non ci sono più colonne da elaborare
    if n == 0
        Q = zeros(m, 0); 
        R = [];
        return;
    end
    
    % Costruzione del riflessore di Householder sulla prima colonna
    x = A(:,1);
    s = -sign(x(1)) * norm(x);
    e1 = zeros(m,1); 
    e1(1) = s;
    v = x - e1;
    
    if norm(v) > 1e-12
        v = v / norm(v);
    else
        v = zeros(m,1);
    end
    
    % Applichiamo la trasformazione H = I - 2vv' ad A senza calcolare H esplicitamente
    % H * A = A - 2 * v * (v' * A)
    A_transf = A - 2 * v * (v' * A);
    
    % Ricorsione sulla sottomatrice inferiore destra
    [Q_new, R_new] = computeThinQR(A_transf(2:end, 2:end));
    
    % 1. Ricostruzione di R (dimensione finale: n x n se m >= n)
    R = [A_transf(1,1), A_transf(1,2:end);
         zeros(n-1,1), R_new];
         
    % 2. Ricostruzione di Q (dimensione finale: m x n)
    % Ricostruiamo la sottomatrice Q_sub aggiungendo la prima riga e colonna della base canonica
    Q_sub = [1, zeros(1, n-1);
             zeros(m-1, 1), Q_new];
             
    % Applichiamo il riflessore corrente H a Q_sub: H * Q_sub = Q_sub - 2 * v * (v' * Q_sub)
    Q = Q_sub - 2 * v * (v' * Q_sub);
end