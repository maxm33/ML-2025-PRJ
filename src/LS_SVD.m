%% ===================================
% LOADING TRAINING DATA (500 patterns)
% ====================================
Dataset_TR = readtable('../data/TR/ML-CUP25-TR.csv');

inputs_TR  = Dataset_TR{:, 2:13};           % 12 inputs
outputs_TR = Dataset_TR{:, 14:end};         % 4 outputs

N = size(inputs_TR,1);

%% ===================================
% I/O NORMALIZATION (zero-mean / unit-variance)
% ====================================

% Inputs
mu_in  = mean(inputs_TR, 1);
std_in = std(inputs_TR, 0, 1);

A = (inputs_TR - mu_in) ./ std_in;

A = [ones(N,1) A];                          % added bias column

% Outputs
mu_out  = mean(outputs_TR, 1);
std_out = std(outputs_TR, 0, 1);

B = (outputs_TR - mu_out) ./ std_out;

%% ===================================
% WEIGHTS CALCULATION
% ====================================

X = pinv(A) * B;

%% ===================================
% PREDICTIONS ON TRAINING DATA
% ====================================

B_pred_norm = A * X;
B_pred = B_pred_norm .* std_out + mu_out;   % denormalized outputs

%% ====================================
% ACCURACY MEASURAMENT
% =====================================

RMSE_norm = sqrt(mean((B - B_pred_norm).^2, 1));
RMSE = sqrt(mean((outputs_TR - B_pred).^2, 1));

disp("RMSE per output (normalized):");
disp(RMSE_norm);

disp("RMSE per output (original scale):");
disp(RMSE);

%% =====================================
% PREDICTIONS ON BLIND TEST DATA (1000 patterns)
% ======================================
%{
Dataset_TS = readtable('../data/TS/ML-CUP25-TS.csv');
inputs_TS = Dataset_TS{:, 2:13};

A_new = (inputs_TS - mu_in) ./ std_in;
A_new = [ones(size(A_new,1),1) A_new];

o_norm = A_new * X;
o = o_norm .* std_out + mu_out;             % denormalized blind predictions
%}
%% ======================================
% VISUALIZATION
% ======================================= 

figure;
scatter3(B(:,1), B(:,2), B(:,3), 40, B(:,4), 'filled');
title('Shape of Training Targets (Normalized)');
xlabel('Output 1'); ylabel('Output 2'); zlabel('Output 3'); colorbar;

figure;
scatter3(B_pred_norm(:,1), B_pred_norm(:,2), B_pred_norm(:,3), 40, B_pred_norm(:,4), 'filled');
title('Predictions on Training Set (Normalized)');
xlabel('Output 1'); ylabel('Output 2'); zlabel('Output 3'); colorbar;

for k = 1:4
    figure; hold on;
    scatter(B(:,k), B_pred_norm(:,k), 40, 'blue', 'filled');
    x = linspace(min(B(:,k)), max(B(:,k)), 200);
    plot(x, x, 'r--', 'LineWidth', 2); 
    xlabel('True Target'); ylabel('Predicted Target');
    title(['Output ' num2str(k) ' - True vs Predicted (Normalized)']);
    grid on; axis equal;
end
