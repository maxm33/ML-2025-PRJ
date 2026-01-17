% Load test inputs
A_test = readmatrix('../data/TS/ML-CUP25-TS.csv');
A_test = A_test(:,2:13);

% Load model
load('./choices/chosen.mat', 'model');

k = numel(model.weights_final);
P = size(A_test,1);

Y_sum = [];

% Activation function
leaky = @(x) max(0.01 * x, x);

for f = 1:k
    % Load final weights of fold
    W1 = model.weights_final(f).W1; b1 = model.weights_final(f).b1;
    W2 = model.weights_final(f).W2; b2 = model.weights_final(f).b2;
    W3 = model.weights_final(f).W3; b3 = model.weights_final(f).b3;

    % Load normalization of fold
    muA  = model.norm(f).muA;  muB  = model.norm(f).muB;
    stdA = model.norm(f).stdA; stdB = model.norm(f).stdB;

    % Normalize inputs
    A_test_norm = (A_test - muA) ./ stdA;

    % Feedforward
    Z1 = W1 * A_test_norm' + b1; H1 = leaky(Z1);
    Z2 = W2 * H1 + b2;           H2 = leaky(Z2);
    Yf = (W3 * H2 + b3)';

    % Denormalize outputs
    Yf_den = Yf .* stdB + muB;

    % Accumulate
    if isempty(Y_sum)
        Y_sum = Yf_den;
    else
        Y_sum = Y_sum + Yf_den;
    end
end

% Average predictions over folds
Y_ensemble = Y_sum / k;

writematrix(Y_ensemble, 'ensemble_predictions.csv');
