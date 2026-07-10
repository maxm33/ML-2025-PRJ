function [bestParams1, bestScore1] = grid_search_deflectedSubgradient_VolumeAndColorTV()
    % Grid Values
    numHidden1_vals = [60]; % 80 il meglio
    numHidden2_vals = [40]; % da capire
    activation_funs  = ["tanh"];
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
    patience        = [200];
    tolerance       = [0.01];
    
    % Combinations
    combo = [];
    for a = numHidden1_vals
        for b = numHidden2_vals
            for c = activation_funs
                for d = lambda_vals
                    for e = beta_vals
                        for f = cg_vals
                            for g = cy_vals
                                for h = cr_vals
                                    for i = tau0_vals
                                        for l = tau_p_vals
                                            for m = tau_f_vals
                                                for n = tau_min_vals
                                                    for o = m_vals
                                                        for p = patience
                                                            for q = tolerance
                                                                combo = [combo; {a, b, c, d, e, f, g, h, i, l, m, n, o, p, q}];
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
    arch_combos = unique(cell2mat(combo(:,1:2)), 'rows');  % coppie uniche (h1, h2)
    
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
        h1      = combo{i,1};
        h2      = combo{i,2};
        fun     = combo{i,3}; 
        lambda  = combo{i,4};
        beta    = combo{i,5};
        cg      = combo{i,6};
        cy      = combo{i,7};
        cr      = combo{i,8};
        tau0    = combo{i,9};
        tau_p   = combo{i,10};
        tau_f   = combo{i,11};
        tau_min = combo{i,12};
        m       = combo{i,13};
        pat     = combo{i,14};
        tol     = combo{i,15};

        key = sprintf('%d_%d', h1, h2);
        w = init_weights(key);
    
        results1(i) = Neural_Network_batch_VolumeAndColorTV(h1, h2, fun, lambda, beta, cg, cy, cr, tau0, tau_p, tau_f, tau_min, m, pat, tol, w);
    end
    
    [bestScore1, bestIdx1] = min(results1);
    bestParams1 = combo(bestIdx1,:);
    
end

function [bestParams1, bestScore1] = grid_search_deflectedSubgradient_VolumeAndSGPTL()
    % Grid Values
    numHidden1_vals = [70]; 
    numHidden2_vals = [40]; 
    activation_fun  = ["tanh"];
    lambda_vals     = [1e-3]; 
    beta_vals       = [1e-2];
    delta_vals      = [1e-1];
    R_vals          = [0.5];
    rho_vals        = [7e-1];
    tau0_vals       = [0.1]; 
    tau_p_vals      = [200]; 
    tau_f_vals      = [0.8]; 
    tau_min_vals    = [1e-5]; 
    m_vals          = [0.5]; 
    patience        = [200];
    tolerance       = [0.01];
    
    % Combinations
    combo = [];
    for a = numHidden1_vals
        for b = numHidden2_vals
            for c = activation_fun
                for d = lambda_vals
                    for e = beta_vals
                        for f = delta_vals
                            for g = R_vals
                                for h = rho_vals
                                    for i = tau0_vals
                                        for l = tau_p_vals
                                            for m = tau_f_vals
                                                for n = tau_min_vals
                                                    for o = m_vals
                                                        for p = patience
                                                            for q = tolerance
                                                                combo = [combo; {a, b, c, d, e, f, g, h, i, l, m, n, o, p, q}];
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
    arch_combos = unique(cell2mat(combo(:,1:2)), 'rows');  % coppie uniche (h1, h2)
    
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
        h1 = combo{i,1};
        h2 = combo{i,2};
        fun = combo{1,3};
        lambda = combo{i,4};
        beta = combo{i, 5};
        delta = combo{i, 6};
        R = combo{i, 7};
        rho = combo{i, 8};
        tau0 = combo{i, 9};
        tau_p = combo{i, 10};
        tau_f = combo{i, 11};
        tau_min = combo{i, 12};
        m = combo{i, 13};
        pat = combo{i, 14};
        tol = combo{i, 15};

        key = sprintf('%d_%d', h1, h2);
        w = init_weights(key);
    
        results1(i) = Neural_Network_batch_VolumeAndSGPTL(h1, h2, fun, lambda, beta, delta, R, rho, tau0, tau_p, tau_f, tau_min, m, pat, tol, w);
    end
    
    [bestScore1, bestIdx1] = min(results1);
    bestParams1 = combo(bestIdx1,:);
    
end

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

%grid_search_deflectedSubgradient_VolumeAndColorTV()
grid_search_deflectedSubgradient_VolumeAndSGPTL()