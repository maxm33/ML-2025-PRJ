data = readmatrix('../data/TR/ML-CUP25-TR.csv');
dataTS = readmatrix('../data/TS/ML-CUP25-TS.csv');

X = data(:, 2:end-4);
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
[n_samples, n_features] = size(X_bias);
Xts_bias = [ones(size(Xts_n,1),1), Xts_n];

n_outputs = size(Yn, 2);

% utilizzo fattorizzazione QR
[Q, R] = computeThinQR(X_bias);
x = R \ (Q' * Yn)
loss = norm(X_bias * x - Yn)^2/n_samples  %MSE
min(Y), max(Y), length(Y)
Y_pred = X_bias * x;

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
    alpha = -sign(x(1)) * norm(x);
    e1 = zeros(m,1); 
    e1(1) = alpha;
    v = x - e1;
    if norm(v) ~= 0 %permette di controllare di non dividere per 0
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
    
    % dimensione della matrice A
    [m, n] = size(A);
    % caso base n = 1
    if n == 1
        x = A(:,1);
        alpha = -sign(x(1)) * norm(x);
        e1 = zeros(m,1);
        e1(1) = alpha;
        v = x - e1;

        H = eye(m) - 2*(v*v')/(v'*v);
        R = alpha;
        Q = H(:,1); % Thin: solo la prima colonna, il resto lo scarto
        return
    end

    % caso ricorsivo
    x_norm = norm(A(:,1)); % norma della prima colonna di A da usare come s
    e = zeros(size(A,1),1); % vettore base
    alpha = -sign(A(1,1)) * x_norm; % segno 
    e(1) = alpha; 
    v = A(:,1) - e; %calcolo v
    I = eye(m);

    H = I - (2 * (v * v') / (v'*v)); %calcolo Householder

    A_new = H * A; 

    [Q_new, R_new] = computeThinQR(A_new(2:end, 2:end));

    R = [alpha, A_new(1,2:end); zeros(n-1,1), R_new];

    q1 = H(:,1);
    Q = [ q1 , H * [ zeros(1,n-1) ; Q_new ] ];
end