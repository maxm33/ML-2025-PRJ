function [bestParams, bestScore] = grid_search_mb()

    % Grid Values
    numHidden1_vals = [10 20 30 40 50 60 70 80 90 100];
    numHidden2_vals = [10 20 30 40 50 60 70 80 90 100];
    eta_vals        = [5e-2 3e-2 1e-2 5e-3 1e-3 5e-4 1e-4 1e-5];
    lambda_vals     = [5e-2 3e-2 1e-2 5e-3 1e-3 5e-4 1e-4 1e-5];
    alpha_vals      = [0.95 0.85 0.75 0.6 0.5 0.4 0.3];
    batch_vals      = [50 125 250 500];

    % Number of combinations
    n1 = numel(numHidden1_vals);
    n2 = numel(numHidden2_vals);
    ne = numel(eta_vals);
    nl = numel(lambda_vals);
    na = numel(alpha_vals);
    nb = numel(batch_vals);

    numCombo = n1*n2*ne*nl*na*nb;
    fprintf('\nTotal combinations: %d\n', numCombo);
    results = zeros(numCombo,1);

    % Start parallel pool
    if isempty(gcp('nocreate'))
        parpool;
    end

    % Progress counter
    dq = parallel.pool.DataQueue;
    afterEach(dq, @updateProgress);
    completed = 0;

    % function for progress update
    function updateProgress(~)
        completed = completed + 1;
        if mod(completed,100) == 0 || completed == numCombo
            percent = 100 * completed / numCombo;
            fprintf('\rCompleted: %d/%d (%.2f%%)', completed, numCombo, percent);
        end
    end
    
    fprintf('\nStarting grid search...\n');
    
    % Parallel grid search
    parfor i = 1:numCombo
    
        % Convert linear index into parameter indices
        [idx_h1, idx_h2, idx_eta, idx_lambda, idx_alpha, idx_batch] = ind2sub([n1 n2 ne nl na nb], i);
    
        % Extract parameters
        h1      = numHidden1_vals(idx_h1);
        h2      = numHidden2_vals(idx_h2);
        eta     = eta_vals(idx_eta);
        lambda  = lambda_vals(idx_lambda);
        alpha   = alpha_vals(idx_alpha);
        batch   = batch_vals(idx_batch);
    
        % Train network
        results(i) = Neural_Network_minibatch(h1, h2, eta, lambda, alpha, batch);
    
        % Notify progress
        send(dq, i);
    end

    % Find best result
    [bestScore, bestIdx] = min(results);

    % Recover best parameters
    [idx_h1, idx_h2, idx_eta, idx_lambda, idx_alpha, idx_batch] = ind2sub([n1 n2 ne nl na nb], bestIdx);

    bestParams = [
        numHidden1_vals(idx_h1), ...
        numHidden2_vals(idx_h2), ...
        eta_vals(idx_eta), ...
        lambda_vals(idx_lambda), ...
        alpha_vals(idx_alpha), ...
        batch_vals(idx_batch)
    ];

    fprintf('\nBest parameters:\n');
    fprintf('Hidden1: %d\n', bestParams(1));
    fprintf('Hidden2: %d\n', bestParams(2));
    fprintf('Eta: %.6f\n', bestParams(3));
    fprintf('Lambda: %.6f\n', bestParams(4));
    fprintf('Alpha: %.2f\n', bestParams(5));
    fprintf('Mini batch: %d\n', bestParams(6));
    fprintf('Best score: %.6f\n', bestScore);
end

grid_search_mb();