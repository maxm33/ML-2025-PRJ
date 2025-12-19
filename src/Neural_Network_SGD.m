%% ===================================
% LOADING TRAINING DATA (500 patterns)
% ====================================
Dataset_TR = readtable('../data/TR/ML-CUP25-TR.csv');

inputs_TR  = Dataset_TR{:, 2:13};
outputs_TR = Dataset_TR{:, 14:end};

N = size(inputs_TR, 2);                     % 12 inputs
M = size(outputs_TR, 2);                    % 4 outputs
P = size(inputs_TR, 1);                     % 500 patterns

% Hyper Parameters
numHidden = 50;
eta = 1e-3;
epochs = 1000;

%% ===================================
% I/O NORMALIZATION (zero-mean / unit-variance)
% ====================================
A = (inputs_TR - mean(inputs_TR,1)) ./ std(inputs_TR,0,1);
B = (outputs_TR - mean(outputs_TR,1)) ./ std(outputs_TR,0,1);

%% ===================================
% NEURAL NETWORK CONFIGURATION (fully connected)
% ====================================

% Input Layer
for i = 1:N
    input_layer(i) = neuron_input_unit(0);
end

% Hidden Layer
for i = 1:numHidden
    hidden_layer(i) = neuron_hidden_unit(generate_hidden_conns_from(input_layer));
end

% Output Layer
for i = 1:M
    output_layer(i) = neuron_output_unit(generate_output_conns_from(hidden_layer));
end

%% ===================================
% BACKPROPAGATION TRAINING LOOP
% ====================================

% Plot Initialization
mse_history = zeros(1, epochs);
figure_handle = figure;
hLine = plot(NaN, NaN, 'b-', 'LineWidth', 2);
xlabel('Epoch'); ylabel('MSE'); title('Learning Curve'); grid on; hold on;

for epoch = 1:epochs
    total_error = 0;
    Yhat = zeros(P, M);
    
    for p = 1:P
        
        % Load pattern p
        for i = 1:N
            input_layer(i).output = A(p,i);  % units forward each input
        end
        
        % Feedforward phase
        for i = 1:numHidden
            hidden_layer(i).compute();
        end
        outputs = zeros(1,M);
        for i = 1:M
            output_layer(i).compute();
            outputs(i) = output_layer(i).output;
            Yhat(p, :) = outputs;
        end

        total_error = total_error + 1/2 * sum((B(p,:) - outputs).^2);

        % Output signals
        output_signals = zeros(1, M);
        for k = 1:M
            output_signals(k) = (B(p,k) - outputs(k)); %* output_layer(k).sigmoid_derivative(outputs(k));
        end
    
        % Hidden signals
        hidden_signals = zeros(1, numHidden);
        for j = 1:numHidden
            summation = 0;
    
            for k = 1:M
                summation = summation + ...
                    output_signals(k) * ...
                    output_layer(k).input_connections(j).weight;
            end
    
            hidden_signals(j) = ...
                summation * hidden_layer(j).Leaky_ReLU_derivative(hidden_layer(j).net);
        end
    
        % Update kj weights
        for k = 1:M
            output_layer(k).bias_weight = ...
                output_layer(k).bias_weight + eta * output_signals(k);
    
            for j = 1:numHidden
                output_layer(k).input_connections(j).weight = ...
                    output_layer(k).input_connections(j).weight + ...
                    eta * output_signals(k) * hidden_layer(j).output;
            end
        end
    
        % Update ji weights
        for j = 1:numHidden
            hidden_layer(j).bias_weight = ...
                hidden_layer(j).bias_weight + eta * hidden_signals(j);
    
            for i = 1:numel(hidden_layer(j).input_connections)
                hidden_layer(j).input_connections(i).weight = ...
                    hidden_layer(j).input_connections(i).weight + ...
                    eta * hidden_signals(j) * ...
                    hidden_layer(j).input_connections(i).neuron.output;
            end
        end
    end

    % Compute total error over an epoch (1/2 factor included)
    mse = total_error / (P * M) * 2;
    
    fprintf('Epoch %d | MSE = %.6f\n', epoch, mse);

    % Collected network outputs over an epoch
    err = B - Yhat;
    mse_per_output = mean(err.^2, 1);
    disp(mse_per_output);

    % Live Plot
    mse_history(epoch) = mean(mse_per_output);
    set(hLine, 'XData', 1:epoch, 'YData', mse_history(1:epoch));
    drawnow;
end
%%
function hidden_conns = generate_hidden_conns_from(input_units)
    hidden_conns(1, numel(input_units)) = struct('neuron', [], 'weight', []);

    for i = 1:numel(input_units)
        hidden_conns(i).neuron = input_units(i);
        hidden_conns(i).weight = randn * 0.1;
    end
end

function output_conns = generate_output_conns_from(hidden_units)
    output_conns(1, numel(hidden_units)) = struct('neuron', [], 'weight', []);

    for i = 1:numel(hidden_units)
        output_conns(i).neuron = hidden_units(i);
        output_conns(i).weight = randn * 0.1;
    end
end
