function [bestParams, bestScore] = grid_search_mb(retraining, filename, fold_bool)
    if nargin < 1, retraining = false; end
    if nargin < 2, filename = ''; end
    if nargin < 3, fold_bool = false; end

    % Grid Values
    numHidden1_vals = [70];
    numHidden2_vals = [50];
    activation_vals = ["leakyrelu"];
    eta_vals        = [1e-2];
    cg_vals         = [50];   
    cy_vals         = [200];   
    cr_vals         = [10];   
    lambda_vals     = [1 1e-1 5e-3];
    alpha_vals      = [0.9];
    batch_vals      = [500];
    patience_vals   = [200];
    tolerance_vals  = [1e-4];
    seed_vals       = [1932];

    % Number of combinations
    n1  = numel(numHidden1_vals);
    n2  = numel(numHidden2_vals);
    naf = numel(activation_vals);
    ne  = numel(eta_vals);
    ncg = numel(cg_vals);
    ncy = numel(cy_vals);
    ncr = numel(cr_vals);
    nl  = numel(lambda_vals);
    na  = numel(alpha_vals);
    nb  = numel(batch_vals);
    np  = numel(patience_vals);
    nt  = numel(tolerance_vals);
    ns  = numel(seed_vals);

    numCombo = n1*n2*naf*ne*ncg*ncy*ncr*nl*na*nb*np*nt*ns;
    fprintf('\nTotal combinations: %d\n', numCombo);
    results = zeros(numCombo, 1);

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

        % Stampa alla prima iterazione, poi ogni 5 minuti (300 sec) o alla fine
        if completed == 1 || elapsed - lastPrint >= 300 || completed == numCombo
            lastPrint = elapsed;
            percent = 100 * completed / numCombo;
            rate = completed / elapsed;
            estimated = (numCombo - completed) / rate;

            fprintf('MiniBatch Progress: %d/%d (%.2f%%) | Elapsed: %.1f min | ETA: %.1f min\n', ...
                completed, numCombo, percent, elapsed/60, estimated/60);
            drawnow('update');
        end
    end

    fprintf('\nStarting grid search...\n');

    N = 12; M = 4;
    [H1, H2] = ndgrid(numHidden1_vals, numHidden2_vals);
    arch_combos = [H1(:), H2(:)];
    init_weights = cell(size(arch_combos, 1), 1);

    % Inizializzazione pesi
    if fold_bool
        for k = 1:size(arch_combos, 1)
            h1_init = arch_combos(k, 1);
            h2_init = arch_combos(k, 2);
            if retraining
                data = load(filename);
                model_sel = data.model;
                for fold = 1:5
                    w.W1{fold} = model_sel.weights_init(fold).W1;
                    w.W2{fold} = model_sel.weights_init(fold).W2;
                    w.W3{fold} = model_sel.weights_init(fold).W3;
                end
                w.b1 = model_sel.weights_init(1).b1;
                w.b2 = model_sel.weights_init(1).b2;
                w.b3 = model_sel.weights_init(1).b3;
            else
                for fold = 1:5
                    if activation_vals(1) == "leakyrelu"
                        w.W1{fold} = initHe(h1_init, N);
                        w.W2{fold} = initHe(h2_init, h1_init);
                        w.W3{fold} = initHe(M, h2_init);
                    elseif activation_vals(1) == "tanh"
                        w.W1{fold} = initXavier(h1_init, N);
                        w.W2{fold} = initXavier(h2_init, h1_init);
                        w.W3{fold} = initXavier(M, h2_init);
                    end
                end
                w.b1 = zeros(h1_init, 1);
                w.b2 = zeros(h2_init, 1);
                w.b3 = zeros(M, 1);
            end
            init_weights{k} = w;
        end
    else
        for k = 1:size(arch_combos, 1)
            h1_init = arch_combos(k, 1);
            h2_init = arch_combos(k, 2);
            if retraining
                data = load(filename);
                model_sel = data.model;
                w.W1 = model_sel.weights_init(1).W1;
                w.W2 = model_sel.weights_init(1).W2;
                w.W3 = model_sel.weights_init(1).W3;
                w.b1 = model_sel.weights_init(1).b1;
                w.b2 = model_sel.weights_init(1).b2;
                w.b3 = model_sel.weights_init(1).b3;
            else
                if activation_vals(1) == "leakyrelu"
                    w.W1 = initHe(h1_init, N);
                    w.W2 = initHe(h2_init, h1_init);
                    w.W3 = initHe(M, h1_init);
                elseif activation_vals(1) == "tanh"
                    w.W1 = initXavier(h1_init, N);
                    w.W2 = initXavier(h2_init, h1_init);
                    w.W3 = initXavier(M, h2_init);
                end
                w.b1 = zeros(h1_init, 1);
                w.b2 = zeros(h2_init, 1);
                w.b3 = zeros(M, 1);
            end
            init_weights{k} = w;
        end
    end

    % Parallel grid search
    parfor i = 1:numCombo
        % Convert linear index into parameter indices
        [idx_h1, idx_h2, idx_af, idx_eta, idx_cg, idx_cy, idx_cr, idx_lambda, ...
         idx_alpha, idx_batch, idx_pat, idx_tol, idx_seed] = ...
            ind2sub([n1 n2 naf ne ncg ncy ncr nl na nb np nt ns], i);

        % Extract parameters
        h1         = numHidden1_vals(idx_h1);
        h2         = numHidden2_vals(idx_h2);
        activation = activation_vals(idx_af);
        eta        = eta_vals(idx_eta);
        green      = cg_vals(idx_cg);
        yellow     = cy_vals(idx_cy);
        red        = cr_vals(idx_cr);
        lambda     = lambda_vals(idx_lambda);
        alpha      = alpha_vals(idx_alpha);
        batch      = batch_vals(idx_batch);
        pat        = patience_vals(idx_pat);
        tol        = tolerance_vals(idx_tol);
        s          = seed_vals(idx_seed);

        arch_idx = find(arch_combos(:,1) == h1 & arch_combos(:,2) == h2);
        w = init_weights{arch_idx};

        % Train network
        results(i) = Neural_Network_minibatch_training(...
            h1, h2, activation, lambda, eta, green, yellow, red, alpha, batch, s, pat, tol, w);

        % Notify progress
        send(dq, i);
    end

    % Find best result
    [bestScore, bestIdx] = min(results);

    [idx_h1, idx_h2, idx_af, idx_eta, idx_cg, idx_cy, idx_cr, idx_lambda, ...
     idx_alpha, idx_batch, idx_pat, idx_tol, idx_seed] = ...
        ind2sub([n1 n2 naf ne ncg ncy ncr nl na nb np nt ns], bestIdx);
    
    bestParams = {
        numHidden1_vals(idx_h1), ...
        numHidden2_vals(idx_h2), ...
        activation_vals(idx_af), ...
        eta_vals(idx_eta), ...
        cg_vals(idx_cg), ...
        cy_vals(idx_cy), ...
        cr_vals(idx_cr), ...
        lambda_vals(idx_lambda), ...
        alpha_vals(idx_alpha), ...
        batch_vals(idx_batch), ...
        patience_vals(idx_pat), ...
        tolerance_vals(idx_tol), ...
        seed_vals(idx_seed)
    };

    fprintf('\nMiglior RMSE (validation): %.6f\n', bestScore);
end

% Inizializzazione Xavier (per tanh)
function W = initXavier(n_out, n_in)
    sigma = sqrt(1 / n_in); 
    W = randn(n_out, n_in) * sigma;
end

% Inizializzazione He (per LeakyReLU)
function W = initHe(n_out, n_in)
    sigma = sqrt(2 / n_in);
    W = randn(n_out, n_in) * sigma;
end

grid_search_mb(1, 'best_lambda0_001', 0)