%% Loading training dataset (500 training examples)
Dataset_TR = readtable('../data/TR/ML-CUP25-TR.csv');

%% Decomposing table
inputs_TR = Dataset_TR(:, 2:13);    % 12 inputs
outputs_TR = Dataset_TR(:, 14:end); % 4 outputs

%% Least Square minimization through Normal Equations

% Input normalization (zero-mean & unit variance)
A_raw = table2array(inputs_TR);
mu_TR = mean(A_raw);
sigma_TR = std(A_raw);
A = (A_raw - mu_TR) ./ sigma_TR;

A = [ones(size(A,1), 1) A];    % adding bias column

% Output normalization
B = table2array(outputs_TR);
B = (B - mean(B)) ./ std(B);

% L2 Regularization (Ridge)
lambda = 0.0001;
I = eye(size(A,2));
I(1,1) = 0;         % no regularization on bias term

X = (A' * A + lambda * I) \ (A' * B);

%% Loading blind dataset (1000 test examples)
Dataset_TS = readtable('../data/TS/ML-CUP25-TS.csv');

%% Extracting test inputs
inputs_TS = Dataset_TS(:, 2:13);
A_new = table2array(inputs_TS);

% Normalization on test inputs
A_new = (A_new - mu_TR) ./ sigma_TR;
A_new = [ones(size(A_new,1), 1) A_new]; % adding bias column

%% Predictions on blind test data
% o = A_new * X
o = A * X;      % now testing on training data accuracy

%% Scatter outputs from training and test for comparison
figure;
scatter3(B(:,1), B(:,2), B(:,3), 40, B(:,4), 'filled');
colorbar;
xlabel('Output 1'); ylabel('Output 2'); zlabel('Output 3');
title('Shape of Training Targets');

figure;
scatter3(o(:,1), o(:,2), o(:,3), 40, o(:,4), 'filled');
colorbar;
xlabel('Output 1'); ylabel('Output 2'); zlabel('Output 3');
%title('Blind Test');
title('Predictions on Training Inputs (after Training)');

figure; hold on;
o_true = B(:,2);
o_pred = o(:,2);
scatter(o_true, o_pred, 40, 'b', 'filled');
% y = x red reference line
x_range = linspace(min([o_true; o_pred]), max([o_true; o_pred]), 200);
plot(x_range, x_range, 'r--', 'LineWidth', 2);
xlabel('True Target'); ylabel('Predicted Target');
title('True vs. Predicted (Output 2)');
grid on;
axis equal;

figure; hold on;
o_true = B(:,4);
o_pred = o(:,4);
scatter(o_true, o_pred, 40, 'b', 'filled');
% y = x red reference line
x_range = linspace(min([o_true; o_pred]), max([o_true; o_pred]), 200);
plot(x_range, x_range, 'r--', 'LineWidth', 2);
xlabel('True Target'); ylabel('Predicted Target');
title('True vs. Predicted (Output 4)');
grid on;
axis equal;

RMSE = sqrt(mean((B - o) .^ 2, 1));
