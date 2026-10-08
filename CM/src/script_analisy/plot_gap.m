function plot_gap_comparison(runs, fbar_lambda, algo_names, colors, out_file, title_str)
% PLOT_GAP_COMPARISON
% runs: struct array with fields .loss_history, .lambda, .algo
% fbar_lambda: containers.Map, lambda -> \bar f_lambda (dalla Tabella 4.1)
% algo_names: cell array dei due algoritmi da confrontare, es {'ColorTV','SGPTL'}
% colors: colori da usare per ciascun algoritmo

    fig = figure('Position', [100 100 700 500]);
    hold on;

    for a = 1:numel(algo_names)
        idx = find(strcmp({runs.algo}, algo_names{a}), 1);
        if isempty(idx)
            continue;
        end
        lh = runs(idx).loss_history;
        lam = runs(idx).lambda;
        fbar = fbar_lambda(lam);

        k = (1:numel(lh))';
        r_k = lh - fbar;

        % Teniamo solo i punti con gap strettamente positivo
        % (necessario per il log-log: log(0) o log(negativo) non è definito)
        valid = r_k > 1e-12;

        loglog(k(valid), r_k(valid), 'Color', colors{a}, 'LineWidth', 1.5, ...
            'DisplayName', algo_names{a});
    end

    % --- Retta di riferimento teorica: pendenza -0.5 ---
    % Ancoriamo la retta al primo punto valido della prima curva,
    % per farla partire dalla stessa regione visiva dei dati
    k_ref = k(valid);
    r_ref_start = r_k(valid);
    k_ref_range = [k_ref(1), k_ref(end)];
    r_ref_range = r_ref_start(1) * (k_ref_range / k_ref_range(1)).^(-0.5);

    loglog(k_ref_range, r_ref_range, 'k--', 'LineWidth', 1.2, ...
        'DisplayName', 'Theoretical O(1/\surd{k}), slope = -0.5');

    hold off;
    xlabel('Epoch $k$ (log scale)', 'Interpreter', 'latex');
    ylabel('Gap $r_k = L_k - \bar f_\lambda$ (log scale)', 'Interpreter', 'latex');
    title(title_str);
    legend('Location', 'best');
    grid on;
    set(gca, 'XScale', 'log', 'YScale', 'log');

    exportgraphics(fig, out_file, 'Resolution', 200);
    close(fig);
end

% Caricate i due run che già usate per Fig. 4.1 (lambda=1e-3)
S1 = load('models/ColorTV_Volume/c.mat'); run1 = S1.model;
S2 = load('models/Gradient/Gradient-h1-70-h2-50-lambda-0.001_e81f4b1a.mat'); run2 = S2.model;

runs = struct('loss_history', {run1.loss_history, run2.loss_history}, ...
              'lambda', {1e-3, 1e-3}, ...
              'algo', {'Deflection', 'Momentum'});

% fbar_lambda dalla vostra Tabella 4.1 (già calcolata)
fbar_map = containers.Map({1e-7,1e-6,1e-5,1e-4,1e-3,1e-2}, ...
                            {0.0022322,0.0023357,0.012385,0.067802,0.29102,0.67067});

plot_gap_comparison(runs, fbar_map, {'Deflection','Momentum'}, {[0 0.4 0.8],[0.8 0.3 0]}, ...
    'gap_deflection_vs_heavyball.png', 'Convergence gap: Volume Algorithm vs Heavy Ball (\lambda=10^{-3})');