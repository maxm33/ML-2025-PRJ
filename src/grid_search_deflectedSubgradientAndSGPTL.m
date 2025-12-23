% Grid Values
numHidden1_vals = [30 40 50 60 70 80];
numHidden2_vals = [30 40 50 60 70 80];
lambda_vals     = [1e-5 1e-6 1e-7 1e-8 1e-9];
beta_vals       = [1.5 1 5e-1 1e-1];
delta_multiplicator_vals = [5e-1 1e-1 5e-1 1e-2];
R_vals          = [1 2 3];
rho_vals        = [5e-1 7e-1];

% Combinations
combo = [];
for a = numHidden1_vals
    for b = numHidden2_vals
        for c = lambda_vals
            for d = beta_vals
                for e = delta_multiplicators_vals
                    for f = R_vals
                        for g = rho_vals
                        combo = [combo; a b c d e g];
                        end
                    end
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
    R = combo(1,6);
    rho = combo(1,7);

    results1(i) = Neural_Network_batch_stepsizeRestrictedSGPTL(h1, h2, lambda, beta, delta, R, rho);
    results2(i) = Neural_Network_batch_deflectionRestrictedSGPTL(h1, h2, lambda, beta, delta, R, rho);
end

[bestScore1, bestIdx1] = min(results1);
bestParams1 = combo(bestIdx1,:);

[bestScore2, bestIdx2] = min(results2);
bestParams2 = combo(bestIdx2,:);
