function gradient = GradientComputation(E_out, A, A1, Z1, A2, Z2, W1, W2, W3, activation_function, lambda)

    % LeakyReLU backward
    dleaky = @(x) (x > 0) + 0.01 * (x <= 0);

    % Layer 3 Gradient
    grad_W3 = E_out' * A2;
    grad_b3 = sum(E_out, 1)';
            
    if activation_function == "leakyReLU"
        dA2 = dleaky(Z2);
    elseif activation_function == "tanh"
        dA2 = 1 - A2.^2;
    end
    
    E_h2 = (E_out * W3) .* dA2;
            
    grad_W2 = E_h2' * A1;
    grad_b2 = sum(E_h2, 1)';
            
    if activation_function == "leakyReLU"
        dA1 = dleaky(Z1);
    elseif activation_function == "tanh"
        dA1 = 1 - A1.^2;
    end

    E_h1 = (E_h2 * W2) .* dA1;
            
    grad_W1 = (E_h1' * A);
    grad_b1 = sum(E_h1, 1)';

    gradient = [
          grad_W1(:) + lambda  * sign(W1(:));
          grad_b1(:);
          grad_W2(:) + lambda * sign(W2(:));
          grad_b2(:);
          grad_W3(:) + lambda * sign(W3(:));
          grad_b3(:)
    ];
end