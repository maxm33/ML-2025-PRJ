clear;
close all;
clc;

[lambda_star, results, computeThinQR] = Least_Squares_QR();

rootDir = fileparts(mfilename('fullpath'));
data = readmatrix(fullfile(rootDir, '..', '..', 'data', 'TR', 'ML-CUP25-TR.csv'));

X = data(:, 2:13);
Y = data(:, 14:17);

d = size(X, 2);
n_outputs = size(Y, 2);

X_mean = mean(X);
X_std = max(std(X), 1e-8);
X_norm = (X - X_mean) ./ X_std;

X_b = [ones(size(X_norm, 1), 1), X_norm];

Y_mean = mean(Y);
Y_std = max(std(Y), 1e-8);
Y_norm = (Y - Y_mean) ./ Y_std;

n_samples = size(X_b, 1);

% Use the lambda selected by M2_model_selection
lambda_fixed = lambda_star;

X_aug = [X_b; sqrt(n_samples * lambda_fixed) * eye(d+1)];
Y_aug = [Y_norm; zeros(d+1, n_outputs)];

% Warm-up run
[~, ~] = computeThinQR(X_aug);

% Actual timing
num_trials = 100;

tic;
for i = 1:num_trials
    [Q, R] = computeThinQR(X_aug);
    theta = R \ (Q' * Y_aug);
end
exec_time_avg = toc / num_trials;

fprintf('\n=== Computational Performance ===\n');
fprintf('Average Execution time (QR + Back Sub): %.6f seconds\n', ...
    exec_time_avg);

%% Orthogonality check of Q
orth_error = norm(Q' * Q - eye(size(Q,2)), 'fro');
fprintf('\n=== Numerical Stability ===\n');
fprintf('Orthogonality error ||Q''Q - I||_F : %e\n', ...
    orth_error);

%% Condition number
lambdas_test = logspace(-4, 3, 50);
cond_normal = zeros(length(lambdas_test), 1);
cond_aug = zeros(length(lambdas_test), 1);

for i = 1:length(lambdas_test)

    lam = lambdas_test(i);

    % Normal equations
    XtX_reg = (X_b' * X_b) + (n_samples * lam * eye(d+1));
    cond_normal(i) = cond(XtX_reg);

    % Augmented matrix
    X_aug_test = [X_b; sqrt(n_samples * lam) * eye(d+1)];
    cond_aug(i) = cond(X_aug_test);
end

%% Condition number plot
figure('Name', 'Condition Number', 'Position', [100, 100, 700, 500]);
loglog(lambdas_test, cond_normal, 'r-', 'LineWidth', 2);
hold on;
loglog(lambdas_test, cond_aug, 'b-', 'LineWidth', 2);
% Mark selected lambda
xline(lambda_star, 'k--', sprintf('\\lambda = %g', lambda_star), 'LabelVerticalAlignment', 'bottom');
grid on;
xlabel('Regularization Parameter (\lambda)');
ylabel('Condition Number \kappa');
legend( ...
    '\kappa(X^T X + n\lambda I) [Normal Eqs]', ...
    '\kappa(\tilde{X}) [Augmented Matrix]', ...
    'Location', 'best');
title('Conditioning: Normal Equations vs Augmented Matrix');
drawnow;

saveas( gcf, fullfile(rootDir, 'M2_Conditioning_Comparison.png'));

%% ================================================================
% Effect of Regularization - Ridge Trace
% ================================================================

theta_all = zeros( length(lambdas_test), d * n_outputs);

for i = 1:length(lambdas_test)

    lam = lambdas_test(i);

    X_aug_test = [X_b; sqrt(n_samples * lam) * eye(d+1)];
    Y_aug_test = [Y_norm; zeros(d+1, n_outputs)];

    % Use the custom Householder QR
    [Q_test, R_test] = computeThinQR(X_aug_test);
    theta_mat = R_test \ (Q_test' * Y_aug_test);

    % Exclude bias terms
    theta_features = theta_mat(2:end, :);
    theta_all(i, :) = theta_features(:)';
end

%% Ridge trace plot
figure( 'Name', 'Ridge Trace', 'Position', [150, 150, 800, 500]);
semilogx( lambdas_test, theta_all, 'LineWidth', 1.5);
hold on;
% Mark selected lambda
xline( lambda_star, 'k--', sprintf('Selected \\lambda = %g', lambda_star), 'LineWidth', 1.5);
grid on;
xlabel('Regularization Parameter (\lambda)');
ylabel('Standardized Weights \theta');
title('Ridge Trace: Parameter Shrinkage due to L2 Penalty');
drawnow;

saveas(gcf, fullfile(rootDir, 'M2_Ridge_Trace.png'));

fprintf('\nAnalysis complete.\n');
