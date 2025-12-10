%% Loading training dataset (500 training examples)
Dataset_TR = readtable('../data/TR/ML-CUP25-TR.csv');

%% Decomposing table
inputs = Dataset_TR(:, 2:13);           % 12 inputs
outputs = Dataset_TR(:, 14:end);        % 4 outputs

%% Relevant stats calculations
overallInputRange = [min(min(inputs{:,:})), max(max(inputs{:,:}))];
overallOutputRange = [min(min(outputs{:,:})), max(max(outputs{:,:}))];
inputRangePerColumn = [min(inputs); max(inputs)];
outputRangePerColumn = [min(outputs); max(outputs)];

%% Examples of random plotting of target 4D-points
figure;
scatter3(outputs.TARGET_1, outputs.TARGET_2, outputs.TARGET_3, 40, outputs.TARGET_4, 'filled');
colorbar;
xlabel('Output 1'); ylabel('Output 2'); zlabel('Output 3');
axis([-40 40 -40 40 -80 80]);
clim([-60 60]);
title('4D Plot of Training Dataset v1');

figure;
scatter3(outputs.TARGET_3, outputs.TARGET_4, outputs.TARGET_1, 40, outputs.TARGET_2, 'filled');
colorbar;
xlabel('Output 3'); ylabel('Output 4'); zlabel('Output 1');
axis([-40 40 -40 40 -80 80]);
clim([-60 60]);
title('4D Plot of Training Dataset v2');
