function [Yhat, A1, Z1, A2, Z2] = Forward(A, W1, b1, W2, b2, W3, b3, activation_function)

    % LeakyReLU Activation function
    leaky = @(x) max(0.01 * x, x);
    
    Z1 = A * W1' + b1'; 
    if strcmpi(activation_function, "tanh")
        A1 = tanh(Z1); 
    elseif strcmpi(activation_function, "leakyReLU")
        A1 = leaky(Z1);
    else
        error('Funzione di attivazione non riconosciuta: %s', activation_function);
    end
            
    Z2 = A1 * W2' + b2';
    if strcmpi(activation_function, "tanh")
        A2 = tanh(Z2); 
    elseif strcmpi(activation_function, "leakyReLU")
        A2 = leaky(Z2);
    else
        error('Funzione di attivazione non riconosciuta: %s', activation_function);
    end
            
    Yhat = A2 * W3' + b3';

end