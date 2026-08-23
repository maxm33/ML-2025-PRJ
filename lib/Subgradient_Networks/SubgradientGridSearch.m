function [bestParams1, bestScore1] = grid_search_deflectedSubgradient_VolumeAndColorTV(retraining, filename)

    % Grid Values
    numHidden1_vals = [70];    %ottimale per ColorTV 
    numHidden2_vals = [50];    %ottimale per ColorTV
    lambda_vals     = [1e-3 1e-4 1e-5];   
    beta_vals       = [1e-3 5e-4 1e-4 5e-5];    % minore di 0.0005 troppo lento, maggiore di 0.002 troppo veloce
    cg_vals         = [175 200 225 250];    % 50 valore ottimo per ora
    cy_vals         = [200 400];    % sembra poco importante, fisso a 400 
    cr_vals         = [3 5 10 20];     % 10 sembra il migliore 
    tau0_vals       = [0.1 0.5 1];      % cambia poco,lo fisso a 1
    tau_p_vals      = [200];    % ben distribuite
    tau_f_vals      = [0.9];    % è uguale
    tau_min_vals    = [1e-5];   % questo è un floor raramente raggiunto
    m_vals          = [0.1 0.05 0.01 0.005];    % da 0.01 in giu
    patience        = [300];
    tolerance       = [1e-4];
    activation_funs = ["tanh"];
    seed            = [1932];

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
    ns  = numel(seed);

    numCombo = n1*n2*na*nl*nb*ncg*ncy*ncr*nt0*ntp*ntf*ntm*nm*np*nt*ns;
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
    
        % Stampa la prima iterazione (così vedi subito che è partito), 
        % poi ogni 5 minuti (300 sec), oppure alla fine.
        if completed == 1 || elapsed - lastPrint >= 300 || completed == numCombo
            lastPrint = elapsed;
            percent = 100 * completed / numCombo;
            rate = completed / elapsed;           
            estimated = (numCombo - completed) / rate;
    
            % Usare \n al posto di \r e forzare drawnow garantisce che il log compaia subito
            fprintf('ColorTV Progress: %d/%d (%.2f%%) | Elapsed: %.1f min | ETA: %.1f min\n', ...
                completed, numCombo, percent, elapsed/60, estimated/60);
            drawnow('update');
        end
    end

    fprintf('\nStarting grid search...\n');

    %rng(42);
    N = 12; M = 4;
    
    [H1, H2] = ndgrid(numHidden1_vals, numHidden2_vals);
    arch_combos = [H1(:), H2(:)];
    
    init_weights = cell(size(arch_combos,1),1);
    
    for k = 1:size(arch_combos,1)

        h1_init = arch_combos(k,1);
        h2_init = arch_combos(k,2);
        if retraining
            data = load(filename);
            model_sel = data.model;

            w.W1 = model_sel.initial_weights.W1;
            w.W2 = model_sel.initial_weights.W2;
            w.W3 = model_sel.initial_weights.W3;
        else     
            for fold = 1:5
                if activation_funs(1) == "leakyrelu"
                    w.W1{fold} = initHe(h1_init,N);
                    w.W2{fold} = initHe(h2_init,h1_init);
                    w.W3{fold} = initHe(M,h2_init);
                elseif activation_funs(1) == "tanh"
                    w.W1{fold} = initXavier(h1_init,N);
                    w.W2{fold} = initXavier(h2_init,h1_init);
                    w.W3{fold} = initXavier(M,h2_init);
                end
            end
        end
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
         idx_m, idx_patience, idx_tolerance, idx_seed] = ...
         ind2sub([n1 n2 na nl nb ncg ncy ncr ...
                  nt0 ntp ntf ntm nm np nt ns],i);

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
        s = seed(idx_seed);

        arch_idx = find(arch_combos(:,1)==h1 & arch_combos(:,2)==h2);
        
        w = init_weights{arch_idx};

        results1(i) = Neural_Network_batch_VolumeAndColorTV(...
            h1,h2,fun,...
            lambda,beta,...
            cg,cy,cr,...
            tau0,tau_p,tau_f,tau_min,...
            m,pat,tol,s,w);

        send(dq,i);
    end

    % Find best result
    [bestScore1,bestIdx1] = min(results1);

    % Recover best parameters
    [idx_h1, idx_h2, idx_fun, idx_lambda, idx_beta,...
     idx_cg, idx_cy, idx_cr,...
     idx_tau0, idx_tau_p, idx_tau_f, idx_tau_min,...
     idx_m, idx_patience, idx_tolerance, idx_seed] = ...
     ind2sub([n1 n2 na nl nb ncg ncy ncr ...
              nt0 ntp ntf ntm nm np nt ns],bestIdx1);

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
        tolerance(idx_tolerance),...
        seed(idx_seed)
    };

    fprintf('Miglior RMSE (validation): %.6f\n', bestScore1);

end

function [bestParams1, bestScore1] = grid_search_deflectedSubgradient_VolumeAndSGPTL(retraining, filename)
    % Grid Values
    numHidden1_vals = [70]; %ottimali per gradiente
    numHidden2_vals = [50]; %ottimali per gradiente
    lambda_vals     = [1e-4 1e-5]; %ottimale per sottogradiente
    beta_vals       = [3e-2]; %ottimo
    delta_vals      = [1e-1]; %ottimo
    R_vals          = [0.05]; %ottimo
    rho_vals        = [7e-1 5e-1]; 
    tau0_vals       = [0.1]; 
    tau_p_vals      = [100]; 
    tau_f_vals      = [0.9]; 
    tau_min_vals    = [1e-5]; 
    m_vals          = [0.1]; 
    patience        = [300];
    tolerance       = [1e-4];
    activation_funs = ["tanh"];
    seed            = [1932];

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
    ns  = numel(seed);

    numCombo = n1*n2*na*nl*nb*nd*nR*nrho*nt0*ntp*ntf*ntm*nm*np*nt*ns;
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
    
        % Stampa la prima iterazione poi ogni 5 minuti (300 sec), oppure alla fine.
        if completed == 1 || elapsed - lastPrint >= 300 || completed == numCombo
            lastPrint = elapsed;
            percent = 100 * completed / numCombo;
            rate = completed / elapsed;           
            estimated = (numCombo - completed) / rate;
    
            % Usare \n al posto di \r e forzare drawnow garantisce che il log compaia subito
            fprintf('SGPTL Progress: %d/%d (%.2f%%) | Elapsed: %.1f min | ETA: %.1f min\n', ...
                completed, numCombo, percent, elapsed/60, estimated/60);
            drawnow('update');
        end
    end

    fprintf('\nStarting grid search...\n');

    N = 12; M = 4;
    
    [H1, H2] = ndgrid(numHidden1_vals, numHidden2_vals);
    arch_combos = [H1(:), H2(:)];
    
    init_weights = cell(size(arch_combos,1),1);
    
    for k = 1:size(arch_combos,1)
        h1_init = arch_combos(k,1);
        h2_init = arch_combos(k,2);
        if retraining
            data = load(filename);
            model_sel = data.model;

            w.W1 = model_sel.initial_weights.W1;
            w.W2 = model_sel.initial_weights.W2;
            w.W3 = model_sel.initial_weights.W3;
        else     
            for fold = 1:5
                if activation_funs(1) == "leakyrelu"
                    w.W1{fold} = initHe(h1_init,N);
                    w.W2{fold} = initHe(h2_init,h1_init);
                    w.W3{fold} = initHe(M,h2_init);
                elseif activation_funs(1) == "tanh"
                    w.W1{fold} = initXavier(h1_init,N);
                    w.W2{fold} = initXavier(h2_init,h1_init);
                    w.W3{fold} = initXavier(M,h2_init);
                end
            end
        end
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
         idx_m,idx_patience,idx_tolerance, idx_seed] = ...
         ind2sub([n1 n2 na nl nb nd nR nrho ...
                  nt0 ntp ntf ntm nm np nt ns],i);

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
        s = seed(idx_seed);

        arch_idx = find(arch_combos(:,1)==h1 & arch_combos(:,2)==h2);
        
        w = init_weights{arch_idx};

        results1(i) = Neural_Network_batch_VolumeAndSGPTL(...
            h1,h2,fun,...
            lambda,beta,...
            delta,R,rho,...
            tau0,tau_p,tau_f,tau_min,...
            m,pat,tol,s,w);

        send(dq,i);
    end

    % Find best result
    [bestScore1,bestIdx1] = min(results1);

    % Recover best parameters
    [idx_h1,idx_h2,idx_fun,...
     idx_lambda,idx_beta,idx_delta,...
     idx_R,idx_rho,...
     idx_tau0,idx_tau_p,idx_tau_f,idx_tau_min,...
     idx_m,idx_patience,idx_tolerance,idx_seed] = ...
     ind2sub([n1 n2 na nl nb nd nR nrho ...
              nt0 ntp ntf ntm nm np nt ns],bestIdx1);

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
        tolerance(idx_tolerance),...
        seed(idx_seed)
        };

    fprintf('Miglior RMSE (validation): %.6f\n', bestScore1);

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

% grid_search_deflectedSubgradient_VolumeAndColorTV(0, 'SGPTL1')
grid_search_deflectedSubgradient_VolumeAndSGPTL(1, 'ColorTV1')