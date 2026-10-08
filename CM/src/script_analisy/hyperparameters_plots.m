% Plot_ValidationCurves_Beta_SGPTL.m
% Genera due grafici al variare di beta (SGPTL), a parita' degli altri parametri:
%   1) curve di RMSE di validazione (velocita' di convergenza)
%   2) oscillazione media della training loss (stabilita')
%
% NOTA: presuppone che Neural_Network_batch_VolumeAndSGPTL salvi nel .mat
% (esattamente come la versione aggiornata di ColorTV) i campi:
%   model.rmse_val_curve   (maxEpochs x k)
%   model.final_epoch      (1 x k)
%   model.mean_oscillation (scalare)
%   model.beta, model.delta, model.R, model.rho
% Se questi campi non sono ancora presenti nei tuoi .mat di SGPTL,
% aggiorna prima Neural_Network_batch_VolumeAndSGPTL come gia' fatto per ColorTV.

rootDir = fileparts(mfilename('fullpath'));
modelsDir = fullfile(rootDir, 'models', 'SGPTL');

%% ===================================
% CONFIGURAZIONE: valori di beta da mostrare + parametri fissati
% ====================================
beta_to_plot = [5e-1 3e-1 1e-1 8e-2 5e-2];

% SGPTL parameters
fixed_beta = 2e-2;
fixed_delta0 = 0.5;
fixed_R      = 1e-1;
fixed_rhoSG  = 0.5;
fixed_green = 100;
fixed_yellow = 200;
fixed_red = 3;
fixed_h1 = 70;
fixed_h2 = 50;

%% ===================================
% CARICAMENTO E FILTRAGGIO DEI MODELLI SALVATI
% ====================================
matFiles = dir(fullfile(modelsDir, 'SGPTL-*.mat'));

colors = lines(length(beta_to_plot));
legendEntries = {};
osc_per_beta0 = nan(1, length(beta_to_plot));

%% ===================================
% FIGURA 1: curve di validazione
% ====================================
figure('Position', [100 100 800 500]);
hold on;

for i = 1:length(beta_to_plot)
    target_beta = beta_to_plot(i);
    found = false;

    for f = 1:length(matFiles)
        S = load(fullfile(matFiles(f).folder, matFiles(f).name), 'model');
        m = S.model;

        if abs(m.beta - fixed_beta) / fixed_beta < 1e-6 && ...
           abs(m.delta - fixed_delta0) / fixed_delta0 < 1e-6 && ...
           abs(m.R - target_beta) / target_beta < 1e-6 && ...
           abs(m.rho - fixed_rhoSG) / fixed_rhoSG < 1e-6 && ...
           m.numHidden1 == fixed_h1 && m.numHidden2 == fixed_h2

            % --- Curva di validazione ---
            curve_matrix = m.loss_history;   % maxEpochs x k
            min_epoch = m.final_epoch;    

            rmse_val_curve = curve_matrix(1:min_epoch, :);
            rmse_val_curve = smoothdata(rmse_val_curve, 'movmean', 15);

            plot(1:min_epoch, rmse_val_curve, 'Color', colors(i,:), 'LineWidth', 1.5);
            legendEntries{end+1} = sprintf('R = %.0e', target_beta);

            found = true;

            osc_per_beta0(i) = m.mean_oscillation;
            break;
        end
    end

    if ~found
        warning('Nessun modello trovato per R = %.0e con i parametri fissati specificati.', target_beta);
    end
end

hold off;
xlabel('Epoch');
ylabel('Loss');
title(sprintf('Training curve overs \\lambda'));
legend(legendEntries, 'Location', 'best');
grid on;
set(gca, 'YScale', 'log');

exportgraphics(gcf, fullfile(rootDir, 'validation_curves_R_sgptl.png'), 'Resolution', 300);

