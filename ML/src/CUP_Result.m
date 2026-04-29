load('models/retrained_SGPTL_final_model.mat')

input_layer  = model.input_layer_final;
hidden_layer1 = model.hidden_layer1_final;
hidden_layer2 = model.hidden_layer2_final;
output_layer  = model.output_layer_final;

Dataset_TS = readtable('../data/TS/ML-CUP25-TS.csv');

A_test = Dataset_TS{:,2:13};

A_test_norm = (A_test - model.mu_A) ./ model.std_A;

P_test = size(A_test_norm,1);
Yhat_norm = zeros(P_test, 4);

for p = 1:P_test
    % Input
    for i = 1:numel(input_layer)
        input_layer(i).output = A_test_norm(p,i);
    end

    % Hidden 1
    for i = 1:numel(hidden_layer1)
        hidden_layer1(i).compute();
    end

    % Hidden 2
    for i = 1:numel(hidden_layer2)
        hidden_layer2(i).compute();
    end

    % Output
    for i = 1:numel(output_layer)
        output_layer(i).compute();
        Yhat_norm(p,i) = output_layer(i).output;
    end
end

Yhat = Yhat_norm .* model.std_B + model.mu_B;

ID = (1:P_test)';

T = table(ID, ...
          Yhat(:,1), Yhat(:,2), Yhat(:,3), Yhat(:,4), ...
          'VariableNames', {'ID','y1','y2','y3','y4'});

writetable(T, 'SGPTL_predictions.csv');
