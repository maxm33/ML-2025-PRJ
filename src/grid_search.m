function [bestParams, bestScore] = grid_search_SGD()
    % Grid Values
    numHidden_vals  = [70 80 90 100];
    eta_vals        = [4e-2 3e-2 2e-2 1e-2 9e-3 7e-3 5e-3];
    lambda_vals     = [1e-5 1e-6 1e-7 1e-8 1e-9];
    
    % Combinations
    combo = [];
    for c = lambda_vals
        for a = numHidden_vals
            for b = eta_vals
                combo = [combo; a b c];
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
        eta = combo(i,2);
        lambda = combo(i,3);
    
        results(i) = Neural_Network_SGD(h1, eta, lambda);
    end
    
    [bestScore, bestIdx] = min(results);
    bestParams = combo(bestIdx,:);
end

function [bestParams, bestScore] = grid_search_batch()
    % Grid Values
    numHidden1_vals = [80 100 70 60];
    numHidden2_vals = [100 80 60 70];
    eta_vals        = [3e-1 2e-1 1e-1 1e-2 9e-2 7e-2 6e-2 4e-2];
    lambda_vals     = [1e-5 1e-3 1e-2 1e-7];
    
    % Combinations
    combo = [];
    for d = lambda_vals
        for b = numHidden2_vals
            for a = numHidden1_vals
                for c = eta_vals
                    combo = [combo; a b c d];
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
    
        results(i) = Neural_Network_batch(h1, h2, eta, lambda);
    end
    
    [bestScore, bestIdx] = min(results);
    bestParams = combo(bestIdx,:);
end

function [bestParams1, bestScore1, bestParams2, bestScore2] = grid_search_deflectedSubgradient()
        % Grid Values
        numHidden1_vals = [30 40 50 60 70 80];
        numHidden2_vals = [30 40 50 60 70 80];
        lambda_vals     = [1e-5 1e-6 1e-7 1e-8 1e-9];
        beta_vals       = [1.5 1 5e-1 1e-1];
        delta_multiplicator_vals = [5e-1 1e-1 5e-1 1e-2];
        
        % Combinations
        combo = [];
        for a = numHidden1_vals
            for b = numHidden2_vals
                for c = lambda_vals
                    for d = beta_vals
                        for e = delta_multiplicator_vals
                            combo = [combo; a b c d e];
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
        results1 = zeros(numCombo,1);
        results2 = zeros(numCombo,1);
        
        parfor i = 1:numCombo
            h1 = combo(i,1);
            h2 = combo(i,2);
            lambda = combo(i,3);
            beta = combo(i,4);
            delta = combo(i,5);
        
            results1(i) = Neural_Network_batch_stepsize_restricted(h1, h2, lambda, beta, delta);
            results2(i) = Neural_Network_batch_deflection_restricted(h1, h2, lambda, beta, delta);
        end
        
        [bestScore1, bestIdx1] = min(results1);
        bestParams1 = combo(bestIdx1,:);
        
        [bestScore2, bestIdx2] = min(results2);
        bestParams2 = combo(bestIdx2,:);
end

function [bestParams1, bestScore1, bestParams2, bestScore2] = grid_search_deflectedSubgradient_SGPTL()
    % Grid Values
    numHidden1_vals = [30 40 50 60 70 80];
    numHidden2_vals = [30 40 50 60 70 80];
    lambda_vals     = [1e-5 1e-6 1e-7 1e-8 1e-9];
    beta_vals       = [1.5 1 5e-1 1e-1];
    delta_multiplicator_vals = [5e-1 1e-1 5e-1 1e-2];
    R_vals          = [1 2 3];
    rho_vals        = [5e-1 7e-1];
    
    % Combinations
    combo = [];
    for a = numHidden1_vals
        for b = numHidden2_vals
            for c = lambda_vals
                for d = beta_vals
                    for e = delta_multiplicator_vals
                        for f = R_vals
                            for g = rho_vals
                                combo = [combo; a b c d e f g];
                            end
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
    results1 = zeros(numCombo,1);
    results2 = zeros(numCombo,1);
    
    parfor i = 1:numCombo
        h1 = combo(i,1);
        h2 = combo(i,2);
        lambda = combo(i,3);
        beta = combo(i,4);
        delta = combo(i,5);
        R = combo(i,6);
        rho = combo(i,7);
    
        results1(i) = Neural_Network_batch_stepsizeRestrictedSGPTL(h1, h2, lambda, beta, delta, R, rho);
        results2(i) = Neural_Network_batch_deflectionRestrictedSGPTL(h1, h2, lambda, beta, delta, R, rho);
    end
    
    [bestScore1, bestIdx1] = min(results1);
    bestParams1 = combo(bestIdx1,:);
    
    [bestScore2, bestIdx2] = min(results2);
    bestParams2 = combo(bestIdx2,:);
end

grid_search_SGD();
%grid_search_batch();
%grid_search_deflectedSubgradient();
%grid_search_deflectedSubgradient_SGPTL();
