%% Script per Analisi Statistica dei Top 20 Modelli - SINCRONIZZATO
clear; clc;

% Percorso della cartella
currentDir = fileparts(mfilename('fullpath'));
folderPath = fullfile(currentDir, '..', '..', 'CM', 'src', 'models', 'Gradient');

% Definizione dei parametri da monitorare (nomi usati nel tuo salvataggio)
paramNames = {'numHidden1', 'numHidden2', 'lambda', 'eta', 'alpha', 'batch_size'};

% Ottieni la lista dei file
fileList = dir(fullfile(folderPath, '*.mat'));
numFiles = length(fileList);

dataCell = cell(numFiles, 1);
validCount = 0;

fprintf('Analisi di %d file...\n', numFiles);

for i = 1:numFiles
    currentFile = fullfile(folderPath, fileList(i).name);
    try
        data = load(currentFile);
        % Verifica l'esistenza di model e rmse_validation
        if isfield(data, 'model') && isfield(data.model, 'rmse_val')
            validCount = validCount + 1;
            
            % Creiamo una struct pulita con i nomi che vogliamo nella tabella finale
            s = struct();
            s.FileName = fileList(i).name;
            s.RMSE_Val = data.model.rmse_val;
            
            % Mappatura esatta dai tuoi campi salvati
            s.numHidden1 = data.model.numHidden1;
            s.numHidden2 = data.model.numHidden2;
            s.lambda     = data.model.lambda;
            s.eta        = data.model.eta;
            s.alpha      = data.model.alpha;
            s.batch_size = data.model.batch_size;
            
            dataCell{validCount} = s;
        end
    catch
        continue;
    end
end

% Rimuovi celle vuote
dataCell = dataCell(1:validCount);

if isempty(dataCell)
    error('Nessun modello trovato. Verifica che la cartella contenga file .mat validi.');
end

resTable = struct2table([dataCell{:}]);

% Ordina per RMSE crescente
resTable = sortrows(resTable, 'RMSE_Val', 'ascend');

% Top 50
numToExtract = min(50, size(resTable, 1));
top50 = resTable(1:numToExtract, :);

fprintf('\n--- TOP %d MODELLI ESTRATTI ---\n', numToExtract);
disp(top50(:, {'FileName', 'RMSE_Val'}));

%% Analisi delle frequenze sui Top 50
fprintf('\n--- ANALISI FREQUENZE NEI TOP %d ---\n', numToExtract);

for i = 1:length(paramNames)
    pName = paramNames{i};
    values = top50.(pName);
    
    % Gestione robusta per trovare i valori più frequenti
    [uniqueVals, ~, idxGroup] = unique(values);
    counts = accumarray(idxGroup, 1);
    
    [maxCount, idxBest] = max(counts);
    mostFrequent = uniqueVals(idxBest);
    
    fprintf('Parametro [%s]:\n', pName);
    for v = 1:length(uniqueVals)
        fprintf('  - Valore %g: %d volte\n', uniqueVals(v), counts(v));
    end
    fprintf('  >> VINCITORE (Moda): %g (%d/50)\n\n', mostFrequent, maxCount);
end