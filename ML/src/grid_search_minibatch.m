function [bestParams, bestScore] = grid_search_mb()
    % Grid Values
    numHidden1_vals = [30];
    numHidden2_vals = [20];
    eta_vals        = [0.0001];
    lambda_vals     = [0.001];
    alpha_vals      = [0.9];
    mb_vals         = [500];
    
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