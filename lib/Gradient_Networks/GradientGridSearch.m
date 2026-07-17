function [bestParams, bestScore] = grid_search_mb()
    % Grid Values
    numHidden1_vals = [5 10 15 20 25 30 40 50 60 70 80 100 120 150];
    numHidden2_vals = [5 10 15 20 25 30 40 50 60 70 80 100 120 150];
    eta_vals        = [5e-2 4e-2 3e-2 2e-2 1e-2 5e-3 1e-3 5e-4 1e-4 1e-5];
    lambda_vals     = [5e-2 4e-2 3e-2 2e-2 1e-2 5e-3 1e-3 5e-4 1e-4 1e-5 1e-6];
    alpha_vals      = [0.95 0.85 0.75 0.6 0.5 0.4 0.3 0.2 0.1 0];
    mb_vals         = [50 100 125 250 500];
    
    % Combinations
    combo = [];
    for e = alpha_vals
        for d = lambda_vals
            for b = numHidden2_vals
                for a = numHidden1_vals
                    for c = eta_vals
                        for f = mb_vals
                            combo = [combo; a b c d e f];
                        end
                    end 
                end
            end
        end
    end
    
    % Start parallel pool
    if isempty(gcp('nocreate'))
        parpool;     
    end
    
    numCombo = size(combo,1);
    results = zeros(numCombo,1);
    
    parfor i = 1:numCombo
        h1 = combo(i,1);
        h2 = combo(i,2);
        eta = combo(i,3);
        lambda = combo(i,4);
        alpha = combo(i,5)
        mb = combo(i,6)
    
        results(i) = Neural_Network_minibatch(h1, h2, eta, lambda, alpha, mb);
    end
    
    [bestScore, bestIdx] = min(results);
    bestParams = combo(bestIdx,:);
end

grid_search_mb();