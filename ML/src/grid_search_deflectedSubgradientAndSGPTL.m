% Grid Values
numHidden1_vals = [30 40]; %valore ottimo
numHidden2_vals = [20 30 40]; %valore ottimo
lambda_vals     = [1e-3 1e-4]; %1e-2 e 1e-3 elimininare
beta_vals       = [3e-1 2e-1 1e-1];
delta_multiplicator_vals = [2e-1 1e-1];
R_vals          = [1 0.5];
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

    results1(i) = Neural_Network_batch_stepsizeRestrictedSGPTL_matrix(h1, h2, lambda, beta, delta, R, rho);
    %results2(i) = Neural_Network_batch_deflectionRestrictedSGPTL(h1, h2, lambda, beta, delta, R, rho);
end

[bestScore1, bestIdx1] = min(results1);
bestParams1 = combo(bestIdx1,:);

[bestScore2, bestIdx2] = min(results2);
bestParams2 = combo(bestIdx2,:);



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

function [bestParams1, bestScore1] = grid_search_deflectedSubgradient_SGPTL()
    % Grid Values
    numHidden1_vals = [70];
    numHidden2_vals = [50];
    lambda_vals     = [1e-2];
    beta_vals       = [2e-1];
    delta_multiplicator_vals = [2e-2];
    R_vals          = [0.5];
    rho_vals        = [7e-1];
    alpha_vals      = [0.9 0.7 0.5 0.3];
    
    % Combinations
    combo = [];
    for a = numHidden1_vals
        for b = numHidden2_vals
            for c = lambda_vals
                for d = beta_vals
                    for e = delta_multiplicator_vals
                        for f = R_vals
                            for g = rho_vals
                                for h = alpha_vals
                                    combo = [combo; a b c d e f g h];
                                end
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
    numRuns = 5; % Numero di esecuzioni per ogni combinazione
    results1 = zeros(numCombo, 1);
    
    parfor i = 1:numCombo
        h1 = combo(i,1);
        h2 = combo(i,2);
        lambda = combo(i,3);
        beta = combo(i,4);
        delta = combo(i,5);
        R = combo(i,6);
        rho = combo(i,7);
        alpha = combo(i,8);
        
        % Array temporaneo per memorizzare i punteggi delle diverse esecuzioni
        runScores = zeros(numRuns, 1);
        
        for r = 1:numRuns
            runScores(r) = Neural_Network_batch_stepsizeRestrictedSGPTL(h1, h2, lambda, beta, delta, R, rho, alpha);
        end
        
        % Prendiamo la mediana per ignorare eventuali outlier (inizializzazioni sfortunate)
        results1(i) = median(runScores);
    end
    
    [bestScore1, bestIdx1] = min(results1);
    bestParams1 = combo(bestIdx1,:);
    
    %[bestScore2, bestIdx2] = min(results2);
    %bestParams2 = combo(bestIdx2,:);
end



%grid_search_deflectedSubgradient();
[bestP1, score1] = grid_search_deflectedSubgradient_SGPTL();