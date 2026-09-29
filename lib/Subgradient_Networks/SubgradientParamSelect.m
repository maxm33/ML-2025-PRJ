%% Script per Analisi Statistica dei Top 20 Modelli - SINCRONIZZATO
clear; clc;

% Percorso della cartella
currentDir = fileparts(mfilename('fullpath'));
%folderPath = fullfile(currentDir, '..', '..', 'CM', 'src', 'models', 'ColorTV_Volume', 'stepsize');
folderPath = fullfile(currentDir, '..', '..', 'CM', 'src', 'models', 'SGPTL', 'stepsize');

% Definizione dei parametri da monitorare (nomi usati nel tuo salvataggio)
%paramNames = {'numHidden1', 'numHidden2', 'lambda', 'beta', 'cg', 'cy', 'cr', 'tau0', 'tau_p', 'tau_f', 'tau_min', 'm'};
paramNames = {'numHidden1', 'numHidden2', 'lambda', 'beta', 'delta', 'R', 'rho', 'tau0', 'tau_p', 'tau_f', 'tau_min', 'm'};

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
        if isfield(data, 'model') && isfield(data.model, 'rmse_validation')
            validCount = validCount + 1;
            
            % Creiamo una struct pulita con i nomi che vogliamo nella tabella finale
            s = struct();
            s.FileName = fileList(i).name;
            s.RMSE_Val = data.model.rmse_validation;
            
            % Mappatura esatta dai tuoi campi salvati
            s.numHidden1 = data.model.numHidden1;
            s.numHidden2 = data.model.numHidden2;
            s.lambda     = data.model.lambda;
            s.beta       = data.model.beta;
            s.delta      = data.model.delta;
            s.R          = data.model.R;
            s.rho        = data.model.rho;
            %s.cg         = data.model.cg;
            %s.cy         = data.model.cy;
            %s.cr         = data.model.cr;
            s.tau0       = data.model.tau0;
            s.tau_p      = data.model.tau_p;
            s.tau_f      = data.model.tau_f;
            s.tau_min    = data.model.tau_min;
            s.m          = data.model.m; % Questo legge model.m
            
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

% Converti in tabella (usando cell2mat su array riga)
resTable = struct2table(cell2mat(dataCell'));

% Ordina per RMSE crescente
resTable = sortrows(resTable, 'RMSE_Val', 'ascend');

% Top 20
numToExtract = min(20, size(resTable, 1));
top20 = resTable(1:numToExtract, :);

fprintf('\n--- TOP %d MODELLI ESTRATTI ---\n', numToExtract);
disp(top20(:, {'FileName', 'RMSE_Val'}));

%% Analisi delle frequenze sui Top 20
fprintf('\n--- ANALISI FREQUENZE NEI TOP %d ---\n', numToExtract);

for i = 1:length(paramNames)
    pName = paramNames{i};
    values = top20.(pName);
    
    % Gestione robusta per trovare i valori più frequenti
    [uniqueVals, ~, idxGroup] = unique(values);
    counts = accumarray(idxGroup, 1);
    
    [maxCount, idxBest] = max(counts);
    mostFrequent = uniqueVals(idxBest);
    
    fprintf('Parametro [%s]:\n', pName);
    for v = 1:length(uniqueVals)
        fprintf('  - Valore %g: %d volte\n', uniqueVals(v), counts(v));
    end
    fprintf('  >> VINCITORE (Moda): %g (%d/20)\n\n', mostFrequent, maxCount);
end