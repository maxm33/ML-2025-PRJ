clear;
clc;

% Cartella contenente i file .mat
folder = 'models/Gradient';   % <-- modifica se necessario

files = dir(fullfile(folder, '*.mat'));

% Struttura per salvare il migliore per ogni lambda
best_results = struct();

for i = 1:length(files)

    filepath = fullfile(files(i).folder, files(i).name);

    % Carica il modello
    S = load(filepath, 'model');

    % Controllo
    if ~isfield(S, 'model')
        fprintf('Saltato: %s (model non presente)\n', files(i).name);
        continue;
    end

    if ~isfield(S.model, 'lambda') || ~isfield(S.model, 'best_loss')
        fprintf('Saltato: %s (lambda o best_loss non presente)\n', files(i).name);
        continue;
    end

    lambda = S.model.lambda;
    best_loss = S.model.best_loss;

    % Nome del campo per il lambda
    lambda_key = matlab.lang.makeValidName(sprintf('lambda_%g', lambda));

    % Se è il primo file per questo lambda
    if ~isfield(best_results, lambda_key)

        best_results.(lambda_key).lambda = lambda;
        best_results.(lambda_key).best_loss = best_loss;
        best_results.(lambda_key).file = files(i).name;

    % Altrimenti confronta il best_loss
    elseif best_loss < best_results.(lambda_key).best_loss

        best_results.(lambda_key).best_loss = best_loss;
        best_results.(lambda_key).file = files(i).name;

    end
end


%% Stampa risultati

fprintf('\n========================================\n');
fprintf('MIGLIOR FILE PER OGNI LAMBDA\n');
fprintf('========================================\n\n');

fields = fieldnames(best_results);

% Ordina per lambda
lambda_values = zeros(length(fields), 1);

for i = 1:length(fields)
    lambda_values(i) = best_results.(fields{i}).lambda;
end

[~, order] = sort(lambda_values);

for i = 1:length(order)

    idx = order(i);
    result = best_results.(fields{idx});

    fprintf('lambda = %.10g\n', result.lambda);
    fprintf('best_loss = %.10f\n', result.best_loss);
    fprintf('file = %s\n\n', result.file);

end

n = length(fields);

lambda_col = zeros(n,1);
loss_col = zeros(n,1);
file_col = strings(n,1);

for i = 1:n
    idx = order(i);

    lambda_col(i) = best_results.(fields{idx}).lambda;
    loss_col(i) = best_results.(fields{idx}).best_loss;
    file_col(i) = best_results.(fields{idx}).file;
end

results_table = table(lambda_col, loss_col, file_col, ...
    'VariableNames', {'Lambda', 'BestLoss', 'File'});

disp(results_table);