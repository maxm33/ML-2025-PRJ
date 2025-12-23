%% ===================================
% LOADING TRAINING DATA (500 patterns)
% ====================================
Dataset_TR = readtable('../data/TR/ML-CUP25-TR.csv');

inputs_TR  = Dataset_TR{:, 2:13};           % 12 inputs
outputs_TR = Dataset_TR{:, 14:end};         % 4 outputs

%% ===================================
% I/O NORMALIZATION (zero-mean / unit-variance)
% ====================================

% Inputs
mu_in  = mean(inputs_TR, 1);
std_in = std(inputs_TR, 0, 1);

A = (inputs_TR - mu_in) ./ std_in;

A = [ones(size(A,1),1) A];                  % added bias column

% Outputs
mu_out  = mean(outputs_TR, 1);
std_out = std(outputs_TR, 0, 1);

B = (outputs_TR - mu_out) ./ std_out;

%% ===================================
% WEIGHTS CALCULATION
% ====================================

% Hyper-Parameters
eta = 1e-6;                                 % learning rate factor
lambda = 1e-8;                              % regularization factor
epochs = 300;

W_online = online_gradient_descent(A,B,1e-6,lambda,epochs);

W_batch = batch_gradient_descent(A,B,1e-3,lambda,epochs);
