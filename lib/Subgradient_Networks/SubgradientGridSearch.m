function [bestParams1, bestScore1] = grid_search_deflectedSubgradient_VolumeAndColorTV()
    % Grid Values
    numHidden1_vals = [60];     % 80 il meglio
    numHidden2_vals = [40];     % da capire
    lambda_vals     = [1e-3];   % poca differenza per ora, non sotto 1e-3
    beta_vals       = [0.1];    % minore di 0.0005 troppo lento, maggiore di 0.002 troppo veloce
    cg_vals         = [100];    % 50 valore ottimo per ora
    cy_vals         = [200];    % sembra poco importante, fisso a 400 
    cr_vals         = [10];     % 10 sembra il migliore 
    tau0_vals       = [1];      % cambia poco,lo fisso a 1
    tau_p_vals      = [200];    % ben distribuite
    tau_f_vals      = [0.9];    % è uguale
    tau_min_vals    = [1e-5];   % questo è un floor raramente raggiunto
    m_vals          = [0.7];    % da 0.01 in giu
    patience        = [200];
    tolerance       = [0.01];
    activation_funs = ["tanh"];

    % Number of combinations
    n1  = numel(numHidden1_vals);
    n2  = numel(numHidden2_vals);
    na  = numel(activation_funs);
    nl  = numel(lambda_vals);
    nb  = numel(beta_vals);
    ncg = numel(cg_vals);
    ncy = numel(cy_vals);
    ncr = numel(cr_vals);
    nt0 = numel(tau0_vals);
    ntp = numel(tau_p_vals);
    ntf = numel(tau_f_vals);
    ntm = numel(tau_min_vals);
    nm  = numel(m_vals);
    np  = numel(patience);
    nt  = numel(tolerance);

    numCombo = n1*n2*na*nl*nb*ncg*ncy*ncr*nt0*ntp*ntf*ntm*nm*np*nt;
    fprintf('\nTotal combinations: %d\n',numCombo);
    results1 = zeros(numCombo,1);

    % Start parallel pool
    if isempty(gcp('nocreate'))
        parpool('local', maxNumCompThreads());    
    end

    % Progress counter
    dq = parallel.pool.DataQueue;
    completed = 0;
    tStart = tic;
    lastPrint = 0;
    afterEach(dq, @updateProgress);
    
    function updateProgress(~)
        completed = completed + 1;
        elapsed = toc(tStart);
    
        % Print every 120 seconds
        if elapsed - lastPrint >= 120 || completed == numCombo
            
            lastPrint = elapsed;
            percent = 100*completed/numCombo;
    
            % Estimate remaining time
            rate = completed/elapsed;           % combinations per second
            estimated = (numCombo-completed)/rate;
    
            fprintf('\rCompleted: %d/%d (%.2f%%) | Elapsed: %.1f min(s) / %.1f hour(s) | ETA: %.1f min(s) / %.1f hour(s)', ...
                completed, numCombo, percent, elapsed/60, elapsed/3600, estimated/60, estimated/3600);
        end
    end

    fprintf('\nStarting grid search...\n');

    rng(42);
    N = 12; M = 4;
    
    arch_combos = unique([numHidden1_vals(:), numHidden2_vals(:)],'rows');
    
    init_weights = cell(size(arch_combos,1),1);
    
    for k = 1:size(arch_combos,1)
        h1_init = arch_combos(k,1);
        h2_init = arch_combos(k,2);
        w.W1 = initXavier(h1_init,N);
        w.W2 = initXavier(h2_init,h1_init);
        w.W3 = initXavier(M,h2_init);
        w.b1 = zeros(h1_init,1);
        w.b2 = zeros(h2_init,1);
        w.b3 = zeros(M,1);
    
        init_weights{k}=w;
    end
    
    parfor i = 1:numCombo

        % Convert linear index into parameter indices
        [idx_h1, idx_h2, idx_fun, idx_lambda, idx_beta,...
         idx_cg, idx_cy, idx_cr,...
         idx_tau0, idx_tau_p, idx_tau_f, idx_tau_min,...
         idx_m, idx_patience, idx_tolerance] = ...
         ind2sub([n1 n2 na nl nb ncg ncy ncr ...
                  nt0 ntp ntf ntm nm np nt],i);

        % Extract parameters
        h1 = numHidden1_vals(idx_h1);
        h2 = numHidden2_vals(idx_h2);
        fun = activation_funs(idx_fun);
        lambda = lambda_vals(idx_lambda);
        beta = beta_vals(idx_beta);
        cg = cg_vals(idx_cg);
        cy = cy_vals(idx_cy);
        cr = cr_vals(idx_cr);
        tau0 = tau0_vals(idx_tau0);
        tau_p = tau_p_vals(idx_tau_p);
        tau_f = tau_f_vals(idx_tau_f);
        tau_min = tau_min_vals(idx_tau_min);
        m = m_vals(idx_m);
        pat = patience(idx_patience);
        tol = tolerance(idx_tolerance);

        arch_idx = find(arch_combos(:,1)==h1 & arch_combos(:,2)==h2);
        
        w = init_weights{arch_idx};

        results1(i) = Neural_Network_batch_VolumeAndColorTV(...
            h1,h2,fun,...
            lambda,beta,...
            cg,cy,cr,...
            tau0,tau_p,tau_f,tau_min,...
            m,pat,tol,w);

        send(dq,i);
    end

    % Find best result
    [bestScore1,bestIdx1] = min(results1);

    % Recover best parameters
    [idx_h1, idx_h2, idx_fun, idx_lambda, idx_beta,...
     idx_cg, idx_cy, idx_cr,...
     idx_tau0, idx_tau_p, idx_tau_f, idx_tau_min,...
     idx_m, idx_patience, idx_tolerance] = ...
     ind2sub([n1 n2 na nl nb ncg ncy ncr ...
              nt0 ntp ntf ntm nm np nt],bestIdx1);

    bestParams1 = {
        numHidden1_vals(idx_h1),...
        numHidden2_vals(idx_h2),...
        activation_funs(idx_fun),...
        lambda_vals(idx_lambda),...
        beta_vals(idx_beta),...
        cg_vals(idx_cg),...
        cy_vals(idx_cy),...
        cr_vals(idx_cr),...
        tau0_vals(idx_tau0),...
        tau_p_vals(idx_tau_p),...
        tau_f_vals(idx_tau_f),...
        tau_min_vals(idx_tau_min),...
        m_vals(idx_m),...
        patience(idx_patience),...
        tolerance(idx_tolerance)
    };
end

function [bestParams1, bestScore1] = grid_search_deflectedSubgradient_VolumeAndSGPTL()
    % Grid Values
    numHidden1_vals = [70]; 
    numHidden2_vals = [40]; 
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
    activation_funs = ["tanh"];

    % Number of combinations
    n1  = numel(numHidden1_vals);
    n2  = numel(numHidden2_vals);
    na  = numel(activation_funs);
    nl  = numel(lambda_vals);
    nb  = numel(beta_vals);
    nd  = numel(delta_vals);
    nR  = numel(R_vals);
    nrho = numel(rho_vals);
    nt0 = numel(tau0_vals);
    ntp = numel(tau_p_vals);
    ntf = numel(tau_f_vals);
    ntm = numel(tau_min_vals);
    nm  = numel(m_vals);
    np  = numel(patience);
    nt  = numel(tolerance);

    numCombo = n1*n2*na*nl*nb*nd*nR*nrho*nt0*ntp*ntf*ntm*nm*np*nt;
    fprintf('\nTotal combinations: %d\n',numCombo);
    results1 = zeros(numCombo,1);

    % Start parallel pool
    if isempty(gcp('nocreate'))
        parpool('local', maxNumCompThreads());    
    end

    % Progress counter
    dq = parallel.pool.DataQueue;
    completed = 0;
    tStart = tic;
    lastPrint = 0;
    afterEach(dq, @updateProgress);
    
    function updateProgress(~)
        completed = completed + 1;
        elapsed = toc(tStart);
    
        % Print every 120 seconds
        if elapsed - lastPrint >= 120 || completed == numCombo
            
            lastPrint = elapsed;
            percent = 100*completed/numCombo;
    
            % Estimate remaining time
            rate = completed/elapsed;           % combinations per second
            estimated = (numCombo-completed)/rate;
    
            fprintf('\rCompleted: %d/%d (%.2f%%) | Elapsed: %.1f min(s) / %.1f hour(s) | ETA: %.1f min(s) / %.1f hour(s)', ...
                completed, numCombo, percent, elapsed/60, elapsed/3600, estimated/60, estimated/3600);
        end
    end

    fprintf('\nStarting grid search...\n');

    rng(42);
    N = 12; M = 4;
    
    arch_combos = unique([numHidden1_vals(:), numHidden2_vals(:)],'rows');
    
    init_weights = cell(size(arch_combos,1),1);
    
    for k = 1:size(arch_combos,1)
        h1_init = arch_combos(k,1);
        h2_init = arch_combos(k,2);
        w.W1 = initXavier(h1_init,N);
        w.W2 = initXavier(h2_init,h1_init);
        w.W3 = initXavier(M,h2_init);
        w.b1 = zeros(h1_init,1);
        w.b2 = zeros(h2_init,1);
        w.b3 = zeros(M,1);
    
        init_weights{k}=w;
    end
    
    parfor i = 1:numCombo

        % Convert linear index into parameter indices
        [idx_h1,idx_h2,idx_fun,...
         idx_lambda,idx_beta,idx_delta,...
         idx_R,idx_rho,...
         idx_tau0,idx_tau_p,idx_tau_f,idx_tau_min,...
         idx_m,idx_patience,idx_tolerance] = ...
         ind2sub([n1 n2 na nl nb nd nR nrho ...
                  nt0 ntp ntf ntm nm np nt],i);

        % Extract parameters
        h1 = numHidden1_vals(idx_h1);
        h2 = numHidden2_vals(idx_h2);
        fun = activation_funs(idx_fun);
        lambda = lambda_vals(idx_lambda);
        beta = beta_vals(idx_beta);
        delta = delta_vals(idx_delta);
        R = R_vals(idx_R);
        rho = rho_vals(idx_rho);
        tau0 = tau0_vals(idx_tau0);
        tau_p = tau_p_vals(idx_tau_p);
        tau_f = tau_f_vals(idx_tau_f);
        tau_min = tau_min_vals(idx_tau_min);
        m = m_vals(idx_m);
        pat = patience(idx_patience);
        tol = tolerance(idx_tolerance);

        arch_idx = find(arch_combos(:,1)==h1 & arch_combos(:,2)==h2);
        
        w = init_weights{arch_idx};

        results1(i) = Neural_Network_batch_VolumeAndSGPTL(...
            h1,h2,fun,...
            lambda,beta,...
            delta,R,rho,...
            tau0,tau_p,tau_f,tau_min,...
            m,pat,tol,w);

        send(dq,i);
    end

    % Find best result
    [bestScore1,bestIdx1] = min(results1);

    % Recover best parameters
    [idx_h1,idx_h2,idx_fun,...
     idx_lambda,idx_beta,idx_delta,...
     idx_R,idx_rho,...
     idx_tau0,idx_tau_p,idx_tau_f,idx_tau_min,...
     idx_m,idx_patience,idx_tolerance] = ...
     ind2sub([n1 n2 na nl nb nd nR nrho ...
              nt0 ntp ntf ntm nm np nt],bestIdx1);

    bestParams1 = {
        numHidden1_vals(idx_h1),...
        numHidden2_vals(idx_h2),...
        activation_funs(idx_fun),...
        lambda_vals(idx_lambda),...
        beta_vals(idx_beta),...
        delta_vals(idx_delta),...
        R_vals(idx_R),...
        rho_vals(idx_rho),...
        tau0_vals(idx_tau0),...
        tau_p_vals(idx_tau_p),...
        tau_f_vals(idx_tau_f),...
        tau_min_vals(idx_tau_min),...
        m_vals(idx_m),...
        patience(idx_patience),...
        tolerance(idx_tolerance)
    };
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