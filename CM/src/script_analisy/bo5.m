function plot_gap_comparison(runs, fbar_lambda, algo_names, colors, out_file, title_str)
    fig = figure('Position', [100 100 700 500]);
    hold on;

    k_first_valid = [];
    r_first_valid = [];

    for a = 1:numel(algo_names)
        idx = find(strcmp({runs.algo}, algo_names{a}), 1);
        if isempty(idx), continue; end
        
        lh = runs(idx).loss_history;
        lam = runs(idx).lambda;
        fbar = fbar_lambda(lam);

        k = (1:numel(lh))';
        r_k = lh - fbar;

        valid = r_k > 1e-12 & (k >= 100);
        k_val = k(valid);
        r_val = r_k(valid);

        loglog(k_val, r_val, 'Color', colors{a}, 'LineWidth', 1.5, ...
            'DisplayName', algo_names{a});

        % Salva i dati del primo algoritmo per agganciare la retta teorica
        if isempty(k_first_valid) && ~isempty(k_val)
            k_first_valid = k_val;
            r_first_valid = r_val;
        end
    end

    % --- Retta teorica ancorata alla fase di decadimento ---
    if ~isempty(k_first_valid)
        % Scegliamo un punto intermedio (es. epoca 10^1 o 10^2) per ancorare la retta
        % anziché l'inizio piatto, così da mostrare meglio il tasso di convergenza
        anchor_idx = min(10, numel(k_first_valid)); 
        k_0 = k_first_valid(anchor_idx);
        r_0 = r_first_valid(anchor_idx);

        k_ref_range = [k_first_valid(1), k_first_valid(end)];
        r_ref_range = r_0 * (k_ref_range / k_0).^(-0.5);

        loglog(k_ref_range, r_ref_range, 'k--', 'LineWidth', 1.2, ...
            'DisplayName', 'Theoretical \mathcal{O}(1/\sqrt{k}), slope = -0.5');
    end

    hold off;
    xlabel('Epoch $k$ (log scale)', 'Interpreter', 'latex');
    ylabel('Gap $r_k = L_k - \bar f_\lambda$ (log scale)', 'Interpreter', 'latex');
    title(title_str, 'Interpreter', 'latex');
    legend('Location', 'southwest', 'Interpreter', 'latex'); % Spostata in basso a sinistra per non coprire i dati
    grid on;
    set(gca, 'XScale', 'log', 'YScale', 'log');

    exportgraphics(fig, out_file, 'Resolution', 300);
    close(fig);
end

% Caricate i due run che già usate per Fig. 4.1 (lambda=1e-3)
S1 = load('models/ColorTV_Volume/ColorTV-h1-70-h2-50-lambda-0.001_7c895384.mat'); run1 = S1.model;
S2 = load('models/SGPTL/SGPTL-h1-70-h2-50-lambda-0.001_faa0cec5.mat'); run2 = S2.model;

runs = struct('loss_history', {run1.loss_history, run2.loss_history}, ...
              'lambda', {1e-3, 1e-3}, ...
              'algo', {'colortv', 'sgptl'});

% fbar_lambda dalla vostra Tabella 4.1 (già calcolata)
fbar_map = containers.Map({1e-7,1e-6,1e-5,1e-4,1e-3,1e-2}, ...
                            {0.0022322,0.0023357,0.012385,0.067802,0.29102,0.67067});

plot_gap_comparison(runs, fbar_map, {'colortv','sgptl'}, {[0 0.4 0.8],[0.8 0.3 0]}, ...
    'gap_colortv_vs_sgptl.png', 'Convergence gap: ColorTV vs SGPTL');