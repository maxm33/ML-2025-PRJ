function [best_config, results_table] = select_best_hyperparams(models_dir)
    % SELECT_BEST_HYPERPARAMS
    % Loads all .mat model files from a directory, computes a composite
    % score (final loss, oscillation, training time), and selects the
    % configuration minimizing the score.
    %
    % Usage:
    %   [best_config, results_table] = select_best_hyperparams('models/ColorTV_Volume/stepsize');

    files = dir(fullfile(models_dir, '*.mat'));
    if isempty(files)
        error('No .mat files found in %s', models_dir);
    end

    n = numel(files);
    L_final   = nan(n,1);
    osc       = nan(n,1);
    time_s    = nan(n,1);
    diverged  = false(n,1);
    filenames = strings(n,1);

    % Store hyperparameters for later inspection (adjust fields as needed)
    beta0_v = nan(n,1); cg_v = nan(n,1); cy_v = nan(n,1); cr_v = nan(n,1);

    for i = 1:n
        data = load(fullfile(files(i).folder, files(i).name));
        m = data.model;
        filenames(i) = string(files(i).name);

        % --- Final training loss ---
        if isfield(m, 'loss_history') && ~isempty(m.loss_history)
            L_final(i) = m.loss_history(end);
        elseif isfield(m, 'best_train_mse')
            L_final(i) = m.best_train_mse;
        else
            warning('No loss field found in %s', files(i).name);
        end

        % --- Divergence check ---
        if isnan(L_final(i)) || isinf(L_final(i)) || L_final(i) > 1e6
            diverged(i) = true;
        end

        % --- Oscillation ---
        if isfield(m, 'mean_oscillation')
            osc(i) = m.mean_oscillation;
        end

        % Exclude also excessive oscillation a priori (tune threshold as needed)
        if ~isnan(osc(i)) && osc(i) > 0.5
            diverged(i) = true;
        end

        % --- Training time ---
        if isfield(m, 'training_time')
            time_s(i) = m.training_time;
        end

        % --- Hyperparameters (adjust field names to match your model struct) ---
        if isfield(m, 'beta'),    beta0_v(i) = m.beta;    end
        if isfield(m, 'cg'),      cg_v(i)    = m.cg;      end
        if isfield(m, 'cy'),      cy_v(i)    = m.cy;      end
        if isfield(m, 'cr'),      cr_v(i)    = m.cr;      end
    end

    %% Exclude diverged / excessively oscillating configurations
    valid = ~diverged & ~isnan(L_final) & ~isnan(osc) & ~isnan(time_s);

    if ~any(valid)
        error('No valid (non-diverged) configurations found.');
    end

    fprintf('Excluded %d/%d configurations (divergence or excessive oscillation).\n', ...
        sum(~valid), n);

    %% Min-max normalization (computed only on valid configurations)
    normalize = @(x) (x - min(x)) ./ (max(x) - min(x) + eps);

    L_hat   = nan(n,1);
    osc_hat = nan(n,1);
    T_hat   = nan(n,1);

    L_hat(valid)   = normalize(L_final(valid));
    osc_hat(valid) = normalize(osc(valid));
    T_hat(valid)   = normalize(time_s(valid));

    %% Composite score (equal weights)
    w_L = 4/6; w_osc = 1/6; w_T = 1/6;
    score = nan(n,1);
    score(valid) = w_L * L_hat(valid) + w_osc * osc_hat(valid) + w_T * T_hat(valid);

    %% Build results table
    results_table = table(filenames, beta0_v, cg_v, cy_v, cr_v, ...
        L_final, osc, time_s, L_hat, osc_hat, T_hat, score, diverged, ...
        'VariableNames', {'File','beta0','cg','cy','cr', ...
        'L_final','oscillation','time_s','L_hat','osc_hat','T_hat','score','diverged'});

    results_table = sortrows(results_table, 'score');

    %% Select best
    [~, best_idx] = min(score);
    best_config = results_table(1, :);   % after sorting, first row is best

    fprintf('\nBest configuration:\n');
    disp(best_config);
end

[best_config, results_table] = select_best_hyperparams('models/SGPTL');

% Salva la tabella completa per ispezione/uso nel report
writetable(results_table, 'hyperparam_selection_results.csv');