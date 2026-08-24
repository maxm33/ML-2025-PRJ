paths = {
    'ColorTV1.mat'
    'ColorTV2.mat'
    'ColorTV3.mat'
};

alpha = 0.90;
eta = 0.007;
batch = 400;

if isempty(gcp('nocreate'))
    parpool;
end

results = nan(numel(paths), 1);

parfor i = 1:numel(paths)

    % Load starting model
    loaded = load(paths{i}, 'model');
    previousModel = loaded.model;

    numHidden1 = previousModel.numHidden1;
    numHidden2 = previousModel.numHidden2;
    lambda = previousModel.lambda;
    initWeights = previousModel.initial_weights;

    results(i) = Neural_Network_minibatch( ...
        numHidden1, numHidden2, "tanh", ...
        eta, lambda, alpha, batch, 1932, ...
        initWeights, []);

    fprintf('\npath %d | alpha = %.2f | Score = %.6f\n', ...
        i, alpha, results(i));
end

%% LOAD VALIDATION CURVES

validation_curves = cell(numel(paths), 1);
currDir = fileparts(mfilename('fullpath'));
modelsDir = fullfile(currDir, '..', '..', 'CM', 'src', 'models', 'Gradient');

for i = 1:numel(paths)

    loaded = load(paths{i}, 'model');
    previousModel = loaded.model;

    numHidden1 = previousModel.numHidden1;
    numHidden2 = previousModel.numHidden2;
    lambda = previousModel.lambda;

    files = dir(fullfile(modelsDir, ...
        sprintf('h1-%d-h2-%d-eta-%g-lambda-%g-alpha-%g-batch-%g_*.mat', ...
        numHidden1, numHidden2, eta, lambda, alpha, batch)));

    if isempty(files)
        warning('Nessun modello trovato per path %d', i);
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

colors = lines(numel(paths));

for i = 1:numel(paths)

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
        'DisplayName', sprintf('Run %d', i));
end

xlabel('Epoch');
ylabel('Validation RMSE');
title('ColorTV vs Heavy Ball');

ylim([0.4 1]);

legend('show', 'Location', 'best');
grid on;
hold off;

exportgraphics(gcf, ...
    'colortv_vs_heavyball.png', ...
    'Resolution', 300);

close(gcf);