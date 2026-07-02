function [W1_bar, W2_bar, W3_bar, b1_bar, b2_bar, b3_bar, f_bar, g_bar, sigma, eps_d, iter_since_tau, tau] = VolumeAlgorithm(W1_bar, W2_bar, W3_bar, b1_bar, b2_bar, b3_bar, W1, W2, W3, b1, b2, b3, f_bar, g_bar, m_ss, loss, g, sigma, eps_d, iter_since_tau, tau, tau_min, tau_f, tau_p, gamma, d_curr)

       % Serious / Null Step
       if f_bar - loss >= m_ss * max(1, abs(f_bar))
        
       % Serious Step
       delta_lambda = [
            reshape(W1 - W1_bar, [], 1);
            reshape(b1 - b1_bar, [], 1);
            reshape(W2 - W2_bar, [], 1);
            reshape(b2 - b2_bar, [], 1);
            reshape(W3 - W3_bar, [], 1);
            reshape(b3 - b3_bar, [], 1)
       ];
       delta_f_bar_val = f_bar - loss;
       sigma = sigma - delta_f_bar_val - delta_lambda' * g_bar;
       sigma = max(sigma, 0);
       % Aggiorna ε dopo SS
       eps_d = eps_d - delta_f_bar_val - delta_lambda' * d_curr;
       eps_d = max(eps_d, 0);
                
       % Aggiorna punto di stabilità
       W1_bar = W1; b1_bar = b1;
       W2_bar = W2; b2_bar = b2;
       W3_bar = W3; b3_bar = b3;
       f_bar = loss;
       g_bar = g;
       else
           % Null Step
           eps_d = eps_d + gamma * (sigma - eps_d);
           eps_d = max(eps_d, 0);
       end

       iter_since_tau = iter_since_tau + 1;
       if iter_since_tau >= tau_p
           tau = max(tau_min, tau * tau_f);
           iter_since_tau = 0;
       end
end