path = 'h1-40-h2-70-eta-0.005-lambda-0.1-alpha-0.75-batch-400_f61f74ed.mat';
loaded = load(path, 'model');
previousModel = loaded.model;

numHidden1 = previousModel.numHidden1;
numHidden2 = previousModel.numHidden2;
eta = previousModel.eta;
lambda = previousModel.lambda;
activation = previousModel.activation;
seed = previousModel.seed;
initWeights = previousModel.weights_init;
normalization = previousModel.norm;

alpha_vals = [0.99 0.9 0.75 0.5 0.0];
batch = 400;

if isempty(gcp('nocreate'))
    parpool;
end

results = nan(numel(alpha_vals), 1);

parfor i = 1:numel(alpha_vals)

    alpha = alpha_vals(i);

    results(i) = Neural_Network_minibatch( ...
        numHidden1, numHidden2, activation, ...
        eta, lambda, alpha, batch, seed, ...
        initWeights, normalization);

    fprintf('\nalpha = %.2f | Score = %.6f\n', ...
        alpha, results(i));
end

%% LOAD VALIDATION CURVES

validation_curves = cell(numel(alpha_vals), 1);
currDir = fileparts(mfilename('fullpath'));
modelsDir = fullfile(currDir, '..', '..', 'CM', 'src', 'models', 'Gradient');

for i = 1:numel(alpha_vals)

    alpha = alpha_vals(i);

    files = dir(fullfile(modelsDir, ...
        sprintf('h1-%d-h2-%d-eta-%g-lambda-%g-alpha-%g-batch-%g_*.mat', ...
        numHidden1, numHidden2, eta, lambda, alpha, batch)));

    if isempty(files)
        warning('Nessun modello trovato per alpha = %.2f', alpha);
        continue;
    end

    [~, idx] = max([files.datenum]);
    modelFile = fullfile(modelsDir, files(idx).name);

    loaded = load(modelFile, 'model');

    validation_curves{i} = loaded.model.rmse_val_curve;
end

%% PLOT

figure('Visible', 'off');
hold on;

colors = lines(numel(alpha_vals));

for i = 1:numel(alpha_vals)

    if isempty(validation_curves{i})
        continue;
    end

    curve = validation_curves{i};

    % Find the last epoch where ALL folds have a valid value
    valid_epochs = all(~isnan(curve), 2);
    last_epoch = find(valid_epochs, 1, 'last');

    if isempty(last_epoch)
        continue;
    end

    % Truncate all folds at the common epoch
    curve = curve(1:last_epoch, :);

    % Average validation curve across folds
    curve = mean(curve, 2);

    plot(1:last_epoch, curve, ...
        'Color', colors(i,:), ...
        'LineWidth', 1.5, ...
        'DisplayName', sprintf('\\alpha = %.2f', alpha_vals(i)));
end

xlabel('Epoch');
ylabel('Validation RMSE');
title('Validation Curves over \alpha | batch=400 (Full Batch)');

ylim([0.4 1]);

legend('show', 'Location', 'best');
grid on;
hold off;

exportgraphics(gcf, ...
    'validation_curves_alpha_batch400.png', ...
    'Resolution', 300);

close(gcf);