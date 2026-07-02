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
        
            results1(i) = Neural_Network_batch_stepsizeRestricted_ColorTV(h1, h2, lambda, beta, delta);
            %results2(i) = Neural_Network_batch_deflection_restricted(h1, h2, lambda, beta, delta);
        end
        
        [bestScore1, bestIdx1] = min(results1);
        bestParams1 = combo(bestIdx1,:);
        
        %[bestScore2, bestIdx2] = min(results2);
        %bestParams2 = combo(bestIdx2,:);
end

function [bestParams1, bestScore1] = grid_search_deflectedSubgradient_SGPTL()
    % Grid Values
    numHidden1_vals = [60];
    numHidden2_vals = [40];
    lambda_vals     = [1e-5];
    beta_vals       = [1e-1];
    delta_multiplicator_vals = [6e-2];
    R_vals          = [0.5];
    rho_vals        = [7e-1];
    alpha_vals      = [0.9];
    
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

    rng(42);
    N = 12; M = 4;
    arch_combos = unique(combo(:,1:2), 'rows');  % coppie uniche (h1, h2)
    
    init_weights = containers.Map();
    for i = 1:size(arch_combos, 1)
        h1 = arch_combos(i,1);
        h2 = arch_combos(i,2);
        key = sprintf('%d_%d', h1, h2);
        w.W1 = initXavier(h1, N);
        w.W2 = initXavier(h2, h1);
        w.W3 = initXavier(M, h2);
        w.b1 = zeros(h1, 1);
        w.b2 = zeros(h2, 1);
        w.b3 = zeros(M,  1);
        init_weights(key) = w;
    end
    
    numCombo = size(combo,1);
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
        key = sprintf('%d_%d', h1, h2);
        w = init_weights(key);

        results1(i) = Neural_Network_batch_stepsizeRestrictedSGPTL(h1, h2, lambda, beta, delta, R, rho, alpha, w);
        
    end
    
    [bestScore1, bestIdx1] = min(results1);
    bestParams1 = combo(bestIdx1,:);
    
    %[bestScore2, bestIdx2] = min(results2);
    %bestParams2 = combo(bestIdx2,:);
end

function [bestParams1, bestScore1] = grid_search_deflectedSubgradient_VolumeAndSGPTL()
    % Grid Values
    numHidden1_vals = [70]; 
    numHidden2_vals = [40]; 
    lambda_vals     = [1e-3]; 
    beta_vals       = [1e-1 1e-2];
    delta_vals      = [1e-1 1e-2];
    R_vals          = [0.1 0.5 1];
    rho_vals        = [5e-1 7e-1];
    tau0_vals       = [0.1 1]; 
    tau_p_vals      = [100 200]; 
    tau_f_vals      = [0.8 0.99]; 
    tau_min_vals    = [1e-5]; 
    m_vals          = [0.5]; 
    StepRes_vals    = [0];
    
    % Combinations
    combo = [];
    for a = numHidden1_vals
        for b = numHidden2_vals
            for c = lambda_vals
                for d = beta_vals
                    for e = delta_vals
                        for f = R_vals
                            for g = rho_vals
                                    for i = tau0_vals
                                        for l = tau_p_vals
                                            for m = tau_f_vals
                                                for n = tau_min_vals
                                                    for o = m_vals
                                                        for p = StepRes_vals
                                                            combo = [combo; a b c d e f g i l m n o p];
                                                        end
                                                    end
                                                end
                                            end
                                        end
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
        parpool('local', maxNumCompThreads());    
    end
    
    numCombo = size(combo,1);
    results1 = zeros(numCombo,1);

    rng(42);
    N = 12; M = 4;
    arch_combos = unique(combo(:,1:2), 'rows');  % coppie uniche (h1, h2)
    
    init_weights = containers.Map();
    for i = 1:size(arch_combos, 1)
        h1 = arch_combos(i,1);
        h2 = arch_combos(i,2);
        key = sprintf('%d_%d', h1, h2);
        w.W1 = initXavier(h1, N);
        w.W2 = initXavier(h2, h1);
        w.W3 = initXavier(M, h2);
        w.b1 = zeros(h1, 1);
        w.b2 = zeros(h2, 1);
        w.b3 = zeros(M,  1);
        init_weights(key) = w;
    end
    
    parfor i = 1:numCombo
        h1 = combo(i,1);
        h2 = combo(i,2);
        lambda = combo(i,3);
        beta = combo(i, 4);
        delta = combo(i, 5);
        R = combo(i, 6);
        rho = combo(i, 7);
        tau0 = combo(i, 8);
        tau_p = combo(i, 9);
        tau_f = combo(i, 10);
        tau_min = combo(i, 11);
        m = combo(i, 12);
        StepRes = combo(i, 13);
        key = sprintf('%d_%d', h1, h2);
        w = init_weights(key);
    
        results1(i) = Neural_Network_batch_VolumeAndSGPTL(h1, h2, lambda, beta, delta, R, rho, tau0, tau_p, tau_f, tau_min, m, StepRes, w);
    end
    
    [bestScore1, bestIdx1] = min(results1);
    bestParams1 = combo(bestIdx1,:);
    
end

function [bestParams1, bestScore1] = grid_search_deflectedSubgradient_ColorTV()
    % Grid Values
    numHidden1_vals = [50 60 70 80];
    numHidden2_vals = [40 50];
    lambda_vals     = [1e-6 5e-6 1e-7];
    alpha_vals      = [0.1 0.2]; 
    beta_vals       = [0.0001 0.0005 0.00005];
    cg_vals         = [15 10 20];
    cy_vals         = [80 60 100];
    cr_vals         = [5 3 10];
    StepRes         = [1 0];
    
    % Combinations
    combo = [];
    for a = numHidden1_vals
        for b = numHidden2_vals
            for c = lambda_vals
                for d = alpha_vals
                    for e = beta_vals
                        for f = cg_vals
                            for g = cy_vals
                                for h = cr_vals
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
    results1 = zeros(numCombo,1);

    rng(42);
    N = 12; M = 4;
    arch_combos = unique(combo(:,1:2), 'rows');  % coppie uniche (h1, h2)
    
    init_weights = containers.Map();
    for i = 1:size(arch_combos, 1)
        h1 = arch_combos(i,1);
        h2 = arch_combos(i,2);
        key = sprintf('%d_%d', h1, h2);
        w.W1 = initXavier(h1, N);
        w.W2 = initXavier(h2, h1);
        w.W3 = initXavier(M, h2);
        w.b1 = zeros(h1, 1);
        w.b2 = zeros(h2, 1);
        w.b3 = zeros(M,  1);
        init_weights(key) = w;
    end
    
    parfor i = 1:numCombo
        h1 = combo(i,1);
        h2 = combo(i,2);
        lambda = combo(i,3);
        alpha = combo(i,4);
        beta = combo(i, 5);
        cg = combo(i, 6);
        cy = combo(i, 7);
        cr = combo(i, 8);
        key = sprintf('%d_%d', h1, h2);
        w = init_weights(key);
    
        results1(i) = Neural_Network_batch_stepsizeRestricted_ColorTV(h1, h2, lambda, alpha, beta, cg, cy, cr, w);
    end
    
    [bestScore1, bestIdx1] = min(results1);
    bestParams1 = combo(bestIdx1,:);
    
end

function [bestParams1, bestScore1] = grid_search_deflectedSubgradient_VolumeAndColorTV()
    % Grid Values
    numHidden1_vals = [60]; % 80 il meglio
    numHidden2_vals = [40]; % da capire
    lambda_vals     = [1e-3]; % poca differenza per ora, non sotto 1e-3
    beta_vals       = [0.1]; %minore di 0.0005 troppo lento, maggiore di 0.002 troppo veloce
    cg_vals         = [100]; %50 valore ottimo per ora
    cy_vals         = [200]; %sembra poco importante, fisso a 400 
    cr_vals         = [10]; %10 sembra il migliore 
    tau0_vals       = [1]; % cambia poco,lo fisso a 1
    tau_p_vals      = [200]; % ben distribuite
    tau_f_vals      = [0.9]; %è uguale
    tau_min_vals    = [1e-5]; %questo è un floor raramente raggiunto
    m_vals          = [0.7]; %da 0.01 in giu
    StepRes_vals    = [0];
    
    % Combinations
    combo = [];
    for a = numHidden1_vals
        for b = numHidden2_vals
            for c = lambda_vals
                    for e = beta_vals
                        for f = cg_vals
                            for g = cy_vals
                                for h = cr_vals
                                    for i = tau0_vals
                                        for l = tau_p_vals
                                            for m = tau_f_vals
                                                for n = tau_min_vals
                                                    for o = m_vals
                                                        for p = StepRes_vals
                                                            combo = [combo; a b c e f g h i l m n o p];
                                                        end
                                                    end
                                                end
                                            end
                                        end
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
        parpool('local', maxNumCompThreads());    
    end
    
    numCombo = size(combo,1);
    results1 = zeros(numCombo,1);

    rng(42);
    N = 12; M = 4;
    arch_combos = unique(combo(:,1:2), 'rows');  % coppie uniche (h1, h2)
    
    init_weights = containers.Map();
    for i = 1:size(arch_combos, 1)
        h1 = arch_combos(i,1);
        h2 = arch_combos(i,2);
        key = sprintf('%d_%d', h1, h2);
        w.W1 = initXavier(h1, N);
        w.W2 = initXavier(h2, h1);
        w.W3 = initXavier(M, h2);
        w.b1 = zeros(h1, 1);
        w.b2 = zeros(h2, 1);
        w.b3 = zeros(M,  1);
        init_weights(key) = w;
    end
    
    parfor i = 1:numCombo
        h1 = combo(i,1);
        h2 = combo(i,2);
        lambda = combo(i,3);
        beta = combo(i, 4);
        cg = combo(i, 5);
        cy = combo(i, 6);
        cr = combo(i, 7);
        tau0 = combo(i, 8);
        tau_p = combo(i, 9);
        tau_f = combo(i, 10);
        tau_min = combo(i, 11);
        m = combo(i, 12);
        StepRes = combo(i, 13);
        key = sprintf('%d_%d', h1, h2);
        w = init_weights(key);
    
        results1(i) = Neural_Network_batch_VolumeAndColorTV(h1, h2, lambda, beta, cg, cy, cr, tau0, tau_p, tau_f, tau_min, m, StepRes, w);
    end
    
    [bestScore1, bestIdx1] = min(results1);
    bestParams1 = combo(bestIdx1,:);
    
end

function [bestParams, bestScore] = grid_search_minibatch()
    % Grid Values
    numHidden1_vals = [60];
    numHidden2_vals = [30];
    eta_vals        = [0.0001];
    lambda_vals     = [0.001];
    alpha_vals      = [0.9]; %momentum
    mb_vals         = [20];
    
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

%grid_search_SGD();
%grid_search_batch();
%grid_search_deflectedSubgradient();
%[bestP1, score1] = grid_search_deflectedSubgradient_SGPTL();
%grid_search_minibatch();
%grid_search_deflectedSubgradient_ColorTV();
grid_search_deflectedSubgradient_VolumeAndColorTV();
%grid_search_deflectedSubgradient_VolumeAndSGPTL();

% Inizializzazione di Xavier (per tahn)
function W = initXavier(n_out, n_in)
    sigma = sqrt(1 / (n_in)); 
    W = randn(n_out, n_in) * sigma;
end

% Inizializzazione He (per ReLU)
function W = initHe(n_out, n_in)
    sigma = sqrt(2 / n_in);
    W = randn(n_out, n_in) * sigma;
end

function Leaky_ReLU = FeedforwardLeakyReLU(x)
    leaky = @(x) max(0.01 * x, x);
end