function [bestParams, bestScore] = grid_search_SGD()
    % Grid Values
    numHidden1_vals = [30];
    numHidden2_vals = [10];
    eta_vals        = [5e-4];
    lambda_vals     = [1e-4];
    alpha_vals      = [0.4];
    
    % Combinations
    combo = [];
    for e = alpha_vals
        for d = lambda_vals
            for b = numHidden2_vals
                for a = numHidden1_vals
                    for c = eta_vals
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
    results = zeros(numCombo,1);
    
    parfor i = 1:numCombo
        h1 = combo(i,1);
        h2 = combo(i,2);
        eta = combo(i,3);
        lambda = combo(i,4);
        alpha = combo(i,5)
    
        results(i) = Neural_Network_SGD(h1, h2, eta, lambda, alpha);
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
            runScores(r) = Neural_Network_batch_stepsizeRestrictedSGPTL_matrix(h1, h2, lambda, beta, delta, R, rho, alpha);
        end
        
        % Prendiamo la mediana per ignorare eventuali outlier (inizializzazioni sfortunate)
        results1(i) = median(runScores);
    end
    
    [bestScore1, bestIdx1] = min(results1);
    bestParams1 = combo(bestIdx1,:);
    
    %[bestScore2, bestIdx2] = min(results2);
    %bestParams2 = combo(bestIdx2,:);
end

function [bestParams1, bestScore1] = grid_search_deflectedSubgradient_ColorTV()
    % Grid Values
    numHidden1_vals = [60 70 80];
    numHidden2_vals = [40 50 60];
    lambda_vals     = [1e-1 1e-2];
    alpha_vals      = [0.5 0.3]; %da 0.5 in su diventa instabile e con valori peggiori
    
    % Combinations
    combo = [];
    for a = numHidden1_vals
        for b = numHidden2_vals
            for c = lambda_vals
                for d = alpha_vals
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
    results1 = zeros(numCombo,1);
    
    parfor i = 1:numCombo
        h1 = combo(i,1);
        h2 = combo(i,2);
        lambda = combo(i,3);
        alpha = combo(i,4);
    
        results1(i) = Neural_Network_batch_stepsizeRestricted_ColorTV(h1, h2, lambda, alpha);
    end
    
    [bestScore1, bestIdx1] = min(results1);
    bestParams1 = combo(bestIdx1,:);
    
end

function [bestParams, bestScore] = grid_search_minibatch()
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
    
    numC mbo = size(combo,1);
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

%grid_search_SGD();
%grid_search_batch();
%grid_search_deflectedSubgradient();
[bestP1, score1] = grid_search_deflectedSubgradient_SGPTL();
%grid_search_minibatch();
%grid_search_deflectedSubgradient_ColorTV();