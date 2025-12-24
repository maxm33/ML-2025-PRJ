% Grid Values
numHidden1_vals = [100 90 80 70];
numHidden2_vals = [70 80 90 100];
eta_vals        = [1e-1 9e-2 8e-2 7e-2 6e-2 2e-1];
lambda_vals     = [1e-5 1e-6 1e-7];

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
