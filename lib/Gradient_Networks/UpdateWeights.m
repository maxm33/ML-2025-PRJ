function [W1, W2, W3, b1, b2, b3, vel_W1, vel_W2, vel_W3, vel_b1, vel_b2, vel_b3] = ...
    UpdateWeights(W1, W2, W3, b1, b2, b3, ...
                  vel_W1, vel_W2, vel_W3, ...
                  vel_b1, vel_b2, vel_b3, ...
                  A_b, B_b, eta, lambda, alpha, ...
                  activation_function)

    % Batch size
    P_b = size(A_b,1);

    %% Forward pass
    [Yhat, A1, Z1, A2, Z2] = Forward( ...
        A_b, W1, b1, W2, b2, W3, b3, activation_function);

    %% Output error
    E_out = Yhat - B_b;

    %% Compute packed gradient
    gradient = GradientComputation( ...
        E_out, A_b, A1, Z1, A2, Z2, ...
        W1, W2, W3, activation_function, lambda);

    %% Unpack gradient vector

    idx = 1;

    % W1
    n = numel(W1);
    dW1 = reshape(gradient(idx:idx+n-1), size(W1));
    idx = idx + n;

    % b1
    n = numel(b1);
    db1 = reshape(gradient(idx:idx+n-1), size(b1));
    idx = idx + n;

    % W2
    n = numel(W2);
    dW2 = reshape(gradient(idx:idx+n-1), size(W2));
    idx = idx + n;

    % b2
    n = numel(b2);
    db2 = reshape(gradient(idx:idx+n-1), size(b2));
    idx = idx + n;

    % W3
    n = numel(W3);
    dW3 = reshape(gradient(idx:idx+n-1), size(W3));
    idx = idx + n;

    % b3
    db3 = reshape(gradient(idx:end), size(b3));

    %% Average gradients over the batch
    dW1 = dW1 / P_b;
    dW2 = dW2 / P_b;
    dW3 = dW3 / P_b;

    db1 = db1 / P_b;
    db2 = db2 / P_b;
    db3 = db3 / P_b;

    %% Momentum updates

    vel_W1 = alpha * vel_W1 - eta * dW1;
    W1 = W1 + vel_W1;

    vel_W2 = alpha * vel_W2 - eta * dW2;
    W2 = W2 + vel_W2;

    vel_W3 = alpha * vel_W3 - eta * dW3;
    W3 = W3 + vel_W3;

    vel_b1 = alpha * vel_b1 - eta * db1;
    b1 = b1 + vel_b1;

    vel_b2 = alpha * vel_b2 - eta * db2;
    b2 = b2 + vel_b2;

    vel_b3 = alpha * vel_b3 - eta * db3;
    b3 = b3 + vel_b3;

end