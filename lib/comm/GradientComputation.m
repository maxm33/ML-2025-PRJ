function gradient = GradientComputation(E_out, A, A1, Z1, A2, Z2, W1, W2, W3, activation_function, lambda)

    % Select activation derivative
    switch lower(activation_function)
        case "leakyrelu"
            derivative = @(Z, ~) (Z > 0) + 0.01 * (Z <= 0);
        case "tanh"
            derivative = @(~, A) 1 - A.^2;
        otherwise
            error('Unrecognized activation function: %s', activation_function);
    end

    % Output layer
    grad_W3 = E_out' * A2;
    grad_b3 = sum(E_out, 1)';

    % Hidden layer 2
    dA2 = derivative(Z2, A2);
    E_h2 = (E_out * W3) .* dA2;

    grad_W2 = E_h2' * A1;
    grad_b2 = sum(E_h2, 1)';

    % Hidden layer 1
    dA1 = derivative(Z1, A1);
    E_h1 = (E_h2 * W2) .* dA1;

    grad_W1 = E_h1' * A;
    grad_b1 = sum(E_h1, 1)';

    % Pack gradients applying L1 regularization
    gradient = [
        grad_W1(:) + lambda * sign(W1(:));
        grad_b1(:);
        grad_W2(:) + lambda * sign(W2(:));
        grad_b2(:);
        grad_W3(:) + lambda * sign(W3(:));
        grad_b3(:)
    ];

end