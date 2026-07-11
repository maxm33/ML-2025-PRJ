function [A_tr_norm, A_vl_norm, B_tr_norm, B_vl_norm, muA, stdA, muB, stdB] = normalize_datasets(A_tr, B_tr, A_vl, B_vl)
    muA = mean(A_tr);
    stdA = max(std(A_tr), 1e-8);
    muB = mean(B_tr);
    stdB = max(std(B_tr), 1e-8);
    
    A_tr_norm = normalize_data(A_tr, muA, stdA);
    A_vl_norm = normalize_data(A_vl, muA, stdA);
    B_tr_norm = normalize_data(B_tr, muB, stdB);
    B_vl_norm = normalize_data(B_vl, muB, stdB);
end

function A_norm = normalize_data(A, mu, std)
    A_norm = (A - mu) ./ std;
end