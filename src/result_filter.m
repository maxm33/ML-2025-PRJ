models_dir = './models';

good_dir      = fullfile(models_dir, 'good_models');
prob_dir      = fullfile(models_dir, 'prob');
discarded_dir = fullfile(models_dir, 'discarded');

if ~exist(good_dir, 'dir'),      mkdir(good_dir);      end
if ~exist(prob_dir, 'dir'),      mkdir(prob_dir);      end
if ~exist(discarded_dir, 'dir'), mkdir(discarded_dir); end

files = dir(fullfile(models_dir, '*.mat'));

good_list = struct( ...
    'name', {}, ...
    'mee', {}, ...
    'rmse', {}, ...
    'h1', {}, ...
    'h2', {}, ...
    'eta', {}, ...
    'lambda', {}, ...
    'alpha', {} ...
);

for k = 1:numel(files)
    matFile = fullfile(models_dir, files(k).name);
    data    = load(matFile);
    model   = data.model;

    [~, baseName, ~] = fileparts(matFile);
    plotFile = fullfile(models_dir, [baseName, '_plot.png']);
    hasPlot  = isfile(plotFile);

    if model.mee_cv_mean <= 19.5 && model.rmse_val_min <= 0.6
        targetDir = good_dir;
    
        entry.name   = files(k).name;
        entry.mee    = model.mee_cv_mean;
        entry.rmse   = model.rmse_val_min;
        entry.h1     = model.numHidden1;
        entry.h2     = model.numHidden2;
        entry.eta    = model.eta;
        entry.lambda = model.lambda;
        entry.alpha  = model.alpha;
    
        good_list(end+1) = entry;

    elseif model.rmse_train_min <= 0.3
        targetDir = prob_dir;

    else
        targetDir = discarded_dir;
    end

    movefile(matFile, targetDir);
    if hasPlot, movefile(plotFile, targetDir); end
end

% ============================
% TOP 20 MODELS
% ============================

if isempty(good_list)
    good_files = dir(fullfile(good_dir, '*.mat'));

    for k = 1:numel(good_files)
        matFile = fullfile(good_dir, good_files(k).name);
        data    = load(matFile);
        model   = data.model;

        entry.name   = good_files(k).name;
        entry.mee    = model.mee_cv_mean;
        entry.rmse   = model.rmse_val_min;
        entry.h1     = model.numHidden1;
        entry.h2     = model.numHidden2;
        entry.eta    = model.eta;
        entry.lambda = model.lambda;
        entry.alpha  = model.alpha;

        good_list(end+1) = entry;
    end
end

if ~isempty(good_list)
    T = struct2table(good_list);

    % Sort: primary = MEE, secondary = RMSE_val
    T = sortrows(T, {'mee', 'rmse'}, {'ascend', 'ascend'});

    nTop = min(20, height(T));

    fprintf('\n===== TOP %d GOOD MODELS =====\n\n', nTop);

    for i = 1:nTop
        fprintf([ ...
            '%2d) %s\n' ...
            '    MEE = %.4f | RMSE_VL = %.4f\n' ...
            '    h1=%d | h2=%d | eta=%g | lambda=%g | alpha=%g\n\n'], ...
            i, ...
            T.name{i}, ...
            T.mee(i), ...
            T.rmse(i), ...
            T.h1(i), T.h2(i), T.eta(i), T.lambda(i), T.alpha(i));
    end
end
