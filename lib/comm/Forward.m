function [Yhat, A1, Z1, A2, Z2] = Forward(A, W1, b1, W2, b2, W3, b3, activation_function)

    % Select activation function
    switch lower(activation_function)
        case "tanh"
            activation = @tanh;
        case "leakyrelu"
            activation = @(x) max(0.01 * x, x);
        otherwise
            error('Unrecognized activation function: %s', activation_function);
    end

    % First hidden layer
    Z1 = A * W1' + b1';
    A1 = activation(Z1);

    % Second hidden layer
    Z2 = A1 * W2' + b2';
    A2 = activation(Z2);

    % Output layer (linear)
    Yhat = A2 * W3' + b3';

end