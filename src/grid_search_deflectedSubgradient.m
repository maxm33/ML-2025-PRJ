% Grid Values
numHidden1_vals = [10 20 30 40 50 60 70 80 90 100];
numHidden2_vals = [10 20 30 40 50 60 70 80 90 100];
lambda_vals     = [1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9];
beta_vals        = [2 1.5 1 5e-1 1e-1];
delta_multiplicator_vals = [5e-1 2e-1 1e-1 5e-1 2e-2 1e-2];

% Combinations
combo = [];
for a = numHidden1_vals
    for b = numHidden2_vals
        for c = lambda_vals
            for d = beta_vals
                for e = delta_multiplicator_vals
                    combo = [combo; a b c d e];
                end
            end
        end
    end
end

% Start parallel pool
if isempty(gcp('nocreate'))
    parpool;     
end

numCombo = size(combo,1);
results1 = zeros(numCombo,1);
results2 = zeros(numCombo,1);

parfor i = 1:numCombo
    h1 = combo(i,1);
    h2 = combo(i,2);
    lambda = combo(i,3);
    beta = combo(i,4);
    delta = combo(1,5);

    results1(i) = Neural_Network_batch_stepsize_restricted(h1, h2, lambda, beta, delta);
    results2(i) = Neural_Network_batch_deflection_restricted(h1, h2, lambda, beta, delta);
end

[bestScore1, bestIdx1] = min(results1);
bestParams1 = combo(bestIdx1,:);

[bestScore2, bestIdx2] = min(results2);
bestParams2 = combo(bestIdx2,:);
