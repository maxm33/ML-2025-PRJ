% Grid Values
numHidden1_vals = [60 50 40];
numHidden2_vals = [40 50 60];
eta_vals        = [1e-1 9e-2 8e-2 7e-2 3e-1 2e-1];
lambda_vals     = [1e-4 1e-5 1e-6];

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
