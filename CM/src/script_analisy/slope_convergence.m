lambdas = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2];
f_stars = [0.0022322, 0.0023357, 0.012385, 0.067802, 0.29102, 0.67067];  % best-known value per lambda

thresholds = [0.10, 0.05, 0.01];   % relative gap thresholds
k_start    = 100;                  % start of the fitting window
end_frac   = 0.95;                 % fit never goes beyond 95% of the run (as stated in the report)
gap_cut    = 0.01;                 % fit stops when the relative gap falls below 1%

% Edit folder/prefix of each algorithm. Check that the second entry is really
% the one you want to label "SGPTL" (in your old script it loaded 'Gradient' files).
algos(1).name = 'ColorTV'; algos(1).folder = 'models/ColorTV_Volume'; algos(1).prefix = 'ColorTV';
algos(2).name = 'Gradient';   algos(2).folder = 'models/Gradient';   algos(2).prefix = 'Gradient';

nL = numel(lambdas); nA = numel(algos); nT = numel(thresholds);
slope = NaN(nL, nA);  R2 = NaN(nL, nA);  k_end = NaN(nL, nA);
iters = NaN(nL, nA, nT);

for i = 1:nL
    for a = 1:nA
        file = trova_file(algos(a).folder, algos(a).prefix, lambdas(i));
        if isempty(file)
            fprintf('%s: file not found for lambda = %g\n', algos(a).name, lambdas(i));
            continue;
        end
        S = load(file, 'model');
        [slope(i,a), R2(i,a), k_end(i,a), iters(i,a,:)] = ...
            analyze_run(S.model.loss_history, f_stars(i), k_start, gap_cut, end_frac, thresholds);
    end
end

%% LaTeX: slope, R2 (and end of the fitting window, for your own check)
fprintf('\n=== SLOPE & R2 ===\n\n');
for i = 1:nL
    fprintf('$10^{%d}$ & %s & %s & %s & %s \\\\\n', round(log10(lambdas(i))), ...
        fmt(slope(i,1),'%.3f'), fmt(R2(i,1),'%.3f'), fmt(slope(i,2),'%.3f'), fmt(R2(i,2),'%.3f'));
end

fprintf('\n=== END OF FITTING WINDOW (k_end) ===\n\n');
for i = 1:nL
    fprintf('$10^{%d}$ & %s & %s \\\\\n', round(log10(lambdas(i))), ...
        fmt(k_end(i,1),'%d'), fmt(k_end(i,2),'%d'));
end

%% LaTeX: iterations to reach relative gap thresholds
fprintf('\n=== ITERATIONS TO RELATIVE GAP ===\n\n');
for i = 1:nL
    fprintf('$10^{%d}$ & %s & %s & %s & %s & %s & %s \\\\\n', round(log10(lambdas(i))), ...
        fmt(iters(i,1,1),'%d'), fmt(iters(i,2,1),'%d'), ...
        fmt(iters(i,1,2),'%d'), fmt(iters(i,2,2),'%d'), ...
        fmt(iters(i,1,3),'%d'), fmt(iters(i,2,3),'%d'));
end

%% ============================================================
function [slope, R2, k_end, iters] = analyze_run(loss_history, f_star, k_start, gap_cut, end_frac, thresholds)
    slope = NaN; R2 = NaN; k_end = NaN;
    iters = NaN(1, numel(thresholds));

    best_curve = cummin(loss_history(:));      % monotonization
    residual   = best_curve - f_star;          % best-known gap
    n  = numel(best_curve);
    r0 = residual(1);
    if r0 <= 0, return; end
    rel_gap = residual / r0;

    % iterations needed to reach each relative threshold
    for j = 1:numel(thresholds)
        idx = find(rel_gap <= thresholds(j), 1, 'first');
        if ~isempty(idx), iters(j) = idx; end
    end

    % fitting window: [k_start, min(95% of the run, first k with relative gap <= gap_cut)]
    k_end = round(end_frac * n);
    idx_cut = find(rel_gap <= gap_cut, 1, 'first');
    if ~isempty(idx_cut), k_end = min(k_end, idx_cut); end
    if k_end <= k_start + 10, k_end = NaN; return; end

    k = (k_start:k_end)';
    r = residual(k_start:k_end);
    valid = r > 0;
    if nnz(valid) < 10, return; end

    log_k = log10(k(valid));
    log_r = log10(r(valid));
    p = polyfit(log_k, log_r, 1);
    SS_res = sum((log_r - polyval(p, log_k)).^2);
    SS_tot = sum((log_r - mean(log_r)).^2);
    slope = p(1);
    R2 = 1 - SS_res / SS_tot;
end

function s = fmt(x, f)
    if isnan(x), s = '--'; else, s = sprintf(f, x); end
end

function filepath = trova_file(cartella, prefisso, target_lam)
    filepath = '';
    files = dir(fullfile(cartella, '*.mat'));
    for k = 1:length(files)
        nome = files(k).name;
        if contains(nome, prefisso)
            tokens = regexp(nome, 'lambda-([\d\.eE-]+)', 'tokens');
            if ~isempty(tokens)
                val = str2double(tokens{1}{1});
                if abs(val - target_lam) < 1e-12
                    filepath = fullfile(files(k).folder, nome);
                    return;
                end
            end
        end
    end
end