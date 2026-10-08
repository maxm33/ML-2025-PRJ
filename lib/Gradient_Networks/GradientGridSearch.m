function [bestParams, bestScore] = grid_search_mb(retraining, model_path, cross_val)
    if nargin < 1, retraining = false; end
    if nargin < 2, model_path = ''; end
    if nargin < 3, cross_val = false; end

    % Grid Values
    numHidden1_vals = [70];
    numHidden2_vals = [50];
    activation_vals = ["leakyrelu"];
    eta_vals        = [5e-5];
    cg_vals         = [100];   
    cy_vals         = [200];   
    cr_vals         = [3];   
    lambda_vals     = [1e-2 1e-3 5e-3 1e-4 1e-5 1e-6 1e-7];
    alpha_vals      = [0.9];
    batch_vals      = [500];
    patience_vals   = [Inf];
    tolerance_vals  = [0];
    maxEpochs_vals  = [40000 80000 160000 200000];
    seed_vals       = [679, 42, 123, 1024, 1932, 2026, 31415, 271828, 161803, 98765, 55555];

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
    nmaxEpochs = numel(maxEpochs_vals);
    ns  = numel(seed_vals);

    numCombo = n1*n2*naf*ne*ncg*ncy*ncr*nl*na*nb*np*nt*nmaxEpochs*ns;
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
    
    model_sel = struct();
    if retraining
        data = load(model_path);
        model_sel = data.model;
    end

    % Parallel grid search
    parfor i = 1:numCombo
        % Convert linear index into parameter indices
        [idx_h1, idx_h2, idx_af, idx_eta, idx_cg, idx_cy, idx_cr, idx_lambda, ...
            idx_alpha, idx_batch, idx_pat, idx_tol, idx_maxEpochs, idx_seed] = ...
                ind2sub([n1 n2 naf ne ncg ncy ncr nl na nb np nt nmaxEpochs ns], i);

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
        maxEpochs  = maxEpochs_vals(idx_maxEpochs);
        s          = seed_vals(idx_seed);

        rng(s, 'twister');
        w = struct();
        
        if cross_val
            if retraining
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
                    if activation == "leakyrelu"
                        w.W1{fold} = initHe(h1, N);
                        w.W2{fold} = initHe(h2, h1);
                        w.W3{fold} = initHe(M, h2);
                    elseif activation == "tanh"
                        w.W1{fold} = initXavier(h1, N);
                        w.W2{fold} = initXavier(h2, h1);
                        w.W3{fold} = initXavier(M, h2);
                    end
                end
                w.b1 = zeros(h1, 1);
                w.b2 = zeros(h2, 1);
                w.b3 = zeros(M, 1);
            end
        else
            if retraining
                w.W1 = model_sel.weights_init(1).W1;
                w.W2 = model_sel.weights_init(1).W2;
                w.W3 = model_sel.weights_init(1).W3;
                w.b1 = model_sel.weights_init(1).b1;
                w.b2 = model_sel.weights_init(1).b2;
                w.b3 = model_sel.weights_init(1).b3;
            else
                if activation == "leakyrelu"
                    w.W1 = initHe(h1, N);
                    w.W2 = initHe(h2, h1);
                    w.W3 = initHe(M, h2);
                elseif activation == "tanh"
                    w.W1 = initXavier(h1, N);
                    w.W2 = initXavier(h2, h1);
                    w.W3 = initXavier(M, h2);
                end
                w.b1 = zeros(h1, 1);
                w.b2 = zeros(h2, 1);
                w.b3 = zeros(M, 1);
            end
        end

        % Train network
        results(i) = Neural_Network_minibatch_training(...
            h1, h2, activation, lambda, eta, green, yellow, red, alpha, batch, s, pat, tol, w, maxEpochs);

        % Notify progress
        send(dq, i);
    end

    % Find best result (f*)
    [bestScore, bestIdx] = min(results);

    % Recover best parameters
    [idx_h1, idx_h2, idx_af, idx_eta, idx_cg, idx_cy, idx_cr, idx_lambda, ...
      idx_alpha, idx_batch, idx_pat, idx_tol, idx_maxEpochs, idx_seed] = ...
          ind2sub([n1 n2 naf ne ncg ncy ncr nl na nb np nt nmaxEpochs ns], bestIdx);

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
        maxEpochs_vals(idx_maxEpochs), ...
        seed_vals(idx_seed)
    };

    fprintf('\nMiglior f* (MSE training): %.6f\n', bestScore);
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

grid_search_mb(1, 'best_lambda0_001');