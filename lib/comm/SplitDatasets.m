function [A_test, B_test, A_rest, B_rest] = split_dataset(inputs_raw, outputs_raw, Ns)
    cv_test = cvpartition(Ns,'HoldOut',0.2);

    idx_test = test(cv_test);
    idx_rest = training(cv_test);
    
    A_test = inputs_raw(idx_test,:);
    B_test = outputs_raw(idx_test,:);
    
    A_rest = inputs_raw(idx_rest,:);
    B_rest = outputs_raw(idx_rest,:);
end