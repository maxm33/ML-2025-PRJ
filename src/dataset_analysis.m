% Loading training dataset (500 training examples)
Dataset = readtable('../data/TR/ML-CUP25-TR.csv')

% Decomposing table
inputs = Dataset(:, 2:13)
outputs = Dataset(:, 14:end)

% Relevant stats calculations
overallMaxInput = max(max(inputs{:, :}))
overallMinInput = min(min(inputs{:, :}))
overallMaxOutput = max(max(outputs{:, :}))
overallMinOutput = min(min(outputs{:, :}))
maxInputValuesPerColumn = max(inputs)
minInputValuesPerColumn = min(inputs)
maxOutputValuesPerColumn = max(outputs)
minOutputValuesPerColumn = min(outputs)

% Examples of random plotting of output 4D-points
figure
scatter3(outputs.TARGET_1, outputs.TARGET_2, outputs.TARGET_3, 40, outputs.TARGET_4, 'filled')
colorbar
xlabel('X'); ylabel('Y'); zlabel('Z')
title('4D Plot of Training Dataset v1')

figure
scatter3(outputs.TARGET_3, outputs.TARGET_4, outputs.TARGET_1, 40, outputs.TARGET_2, 'filled')
colorbar
xlabel('X'); ylabel('Y'); zlabel('Z')
title('4D Plot of Training Dataset v2')
