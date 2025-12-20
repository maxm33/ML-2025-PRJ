% Grid Values
numHidden1_vals = [10 20 30 40 50 60 70 80 90 100];
numHidden2_vals = [10 20 30 40 50 60 70 80 90 100];
eta_vals        = [5e-1 4e-1 3e-1 2e-1 1e-1 5e-2 4e-2 3e-2 2e-2 1e-2 5e-3 4e-3 3e-3 2e-3 1e-3];
lambda_vals     = [1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9];

% Combinations
combo = [];
for a = numHidden1_vals
    for b = numHidden2_vals
        for c = eta_vals
            for d = lambda_vals
                combo = [combo; a b c d];
            end
        end
    end
end

% Start parallel pool
if isempty(gcp('nocreate'))
    parpool;     
end

numCombo = size(combo,1);
results = zeros(numCombo,1);

parfor i = 1:numCombo
    h1 = combo(i,1);
    h2 = combo(i,2);
    eta = combo(i,3);
    lambda = combo(i,4);

    results(i) = Neural_Network_batch(h1, h2, eta, lambda);
end

[bestScore, bestIdx] = min(results);
bestParams = combo(bestIdx,:);
