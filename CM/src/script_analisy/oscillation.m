% Plot_TrainingLoss_Zoom_ColorTV.m
% Genera un grafico con zoom sulla parte finale della training loss per la
% configurazione ColorTV finale, per mostrare chiaramente l'oscillazione
% (non-monotonicita') attesa dalla teoria dei metodi subgradiente.

rootDir = fileparts(mfilename('fullpath'));
modelsDir = fullfile(rootDir, 'models', 'Gradient');

%% ===================================
% CONFIGURAZIONE: parametri della configurazione finale scelta
% ====================================
final_beta0 = 1e-2;
final_lambda = 1e-4;
final_cg    = 50;
final_cy    = 200;
final_cr    = 10;
fixed_h1    = 70;
fixed_h2    = 50;

zoom_fraction = 0.25;  % mostra l'ultimo 25% delle epoche (fino all'early stop)

%% ===================================
% CARICAMENTO DEL MODELLO CORRISPONDENTE
% ====================================
matFiles = dir(fullfile(modelsDir, 'Gradient-*.mat'));

m = [];
for f = 1:length(matFiles)
    S = load(fullfile(matFiles(f).folder, matFiles(f).name), 'model');
    cand = S.model;

    if abs(cand.eta - final_beta0) / final_beta0 < 1e-6 && ...
       cand.lambda == final_lambda && ...
       cand.numHidden1 == fixed_h1 && cand.numHidden2 == fixed_h2
        m = cand;
        break;
    end
end

if isempty(m)
    error('Nessun modello trovato per la configurazione finale specificata.');
end

%% ===================================
% ESTRAZIONE DELLA CURVA DI TRAINING LOSS DI UN SINGOLO FOLD
% ====================================
% NOTA: la media tra i k fold smussa l'oscillazione (fold diversi oscillano
% in epoche diverse), nascondendo visivamente il fenomeno che vogliamo
% mostrare. Per l'ispezione visiva usiamo quindi un singolo fold; la
% metrica quantitativa di oscillazione (m.mean_oscillation) resta invece
% correttamente calcolata fold-per-fold e poi mediata.
curve_matrix = m.loss_history;   % maxEpochs x k

this_final_epoch = length(m.loss_history);

rmse_train_curve = curve_matrix(1:this_final_epoch);

% Porzione finale da mostrare nello zoom
zoom_start = round(this_final_epoch * (1 - zoom_fraction));
zoom_range = zoom_start:this_final_epoch;

%% ===================================
% FIGURA CON DUE PANNELLI: curva completa (con box zoom) + delta epoca-su-epoca
% ====================================
figure('Position', [100 100 1000 450]);

% Pannello sinistro: curva completa (scala log), con box che indica la zona di zoom
subplot(1,2,1);
plot(1:this_final_epoch, rmse_train_curve, 'Color', [0 0.4470 0.7410], 'LineWidth', 1.2);
hold on;
yl = ylim;
patch([zoom_start zoom_start this_final_epoch this_final_epoch], ...
      [yl(1) yl(2) yl(2) yl(1)], ...
      [1 0 0], 'FaceAlpha', 0.08, 'EdgeColor', 'r', 'LineStyle', '--');
hold off;
xlabel('Epoch');
ylabel('Training Loss (normalized)');
set(gca, 'YScale', 'log');
grid on;

% Pannello destro: DIFFERENZA epoca-su-epoca nella porzione finale.
% Questo rende visibile l'oscillazione anche quando e' piccola rispetto
% al trend generale di discesa, che qui viene rimosso.
subplot(1,2,2);
zoom_diffs_range = zoom_start:(this_final_epoch-1); % diff ha un elemento in meno
deltas = diff(rmse_train_curve(zoom_start:this_final_epoch));

is_increase = deltas > 0;
bar_colors = repmat([0 0.4470 0.7410], length(deltas), 1);
bar_colors(is_increase, :) = repmat([0.8500 0.3250 0.0980], sum(is_increase), 1);

b = bar(zoom_diffs_range, deltas, 1, 'FaceColor', 'flat');
b.CData = bar_colors;
yline(0, 'k-', 'LineWidth', 0.8);
xlabel('Epoch');
ylabel('\Delta Loss');
title(sprintf('Epoch-over-epoch variation', ...
    100 * sum(is_increase) / length(deltas)));
grid on;

exportgraphics(gcf, fullfile(rootDir, 'training_loss_zoom_heavyball.png'), 'Resolution', 300);

%% Stampa un riepilogo numerico per il testo del report
frac_increasing = sum(is_increase) / length(deltas);
fprintf('Frazione di epoche con aumento della loss (ultimo %.0f%% del training, fold %d): %.4f\n', ...
    zoom_fraction*100, fold_to_plot, frac_increasing);