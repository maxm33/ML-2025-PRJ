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
numHidden1 = 50;
numHidden2 = 50;
eta = 0.4;
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

% Hidden Layer 1
for i = 1:numHidden1
    hidden_layer1(i) = neuron_hidden_unit(generate_hidden_conns_from(input_layer));
end

% Hidden Layer 2
for i = 1:numHidden2
    hidden_layer2(i) = neuron_hidden_unit(generate_hidden_conns_from(hidden_layer1));
end

% Output Layer
for i = 1:M
    output_layer(i) = neuron_output_unit(generate_output_conns_from(hidden_layer2));
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

    % Gradient Accumulators
    grad_W_h1 = cell(1,numHidden1); grad_b_h1 = zeros(1,numHidden1);
    grad_W_h2 = cell(1,numHidden2); grad_b_h2 = zeros(1,numHidden2);
    grad_W_out = zeros(M,numHidden2); grad_b_out = zeros(1,M);

    for j = 1:numHidden1
        grad_W_h1{j} = zeros(1,N);
    end
    for j = 1:numHidden2
        grad_W_h2{j} = zeros(1,numHidden1);
    end

    % Loop over all patterns
    for p = 1:P
        %% Feedforward phase
        for i = 1:N
            input_layer(i).output = A(p,i); % load pattern p
        end
        for i = 1:numHidden1
            hidden_layer1(i).compute();
        end
        for i = 1:numHidden2
            hidden_layer2(i).compute();
        end
        outputs = zeros(1,M);
        for i = 1:M
            output_layer(i).compute();
            outputs(i) = output_layer(i).output;
            Yhat(p,:) = outputs;
        end

        total_error = total_error + 1/2 * sum((B(p,:) - outputs).^2);

        %% BackPropagation phase

        % Output signals
        output_signals = zeros(1,M);
        for k = 1:M
            output_signals(k) = (B(p,k) - outputs(k)); %* output_layer(k).sigmoid_derivative(outputs(k));
        end

        % Hidden layer 2 signals
        hidden2_signals = zeros(1, numHidden2);
        for j = 1:numHidden2
            summation = 0;
            for k = 1:M
                summation = summation + output_signals(k) * output_layer(k).input_connections(j).weight;
            end
            hidden2_signals(j) = summation * hidden_layer2(j).Leaky_ReLU_derivative(hidden_layer2(j).net);
        end

        % Hidden layer 1 signals
        hidden1_signals = zeros(1, numHidden1);
        for j = 1:numHidden1
            summation = 0;
            for u = 1:numHidden2
                summation = summation + hidden2_signals(u) * hidden_layer2(u).input_connections(j).weight;
            end
            hidden1_signals(j) = summation * hidden_layer1(j).Leaky_ReLU_derivative(hidden_layer1(j).net);
        end

        %% Gradients Accumulation

        % Output layer
        for k = 1:M
            grad_b_out(k) = grad_b_out(k) + output_signals(k);
            for j = 1:numHidden2
                grad_W_out(k,j) = grad_W_out(k,j) + output_signals(k) * hidden_layer2(j).output;
            end
        end

        % Hidden layer 2
        for j = 1:numHidden2
            grad_b_h2(j) = grad_b_h2(j) + hidden2_signals(j);
            for i = 1:numHidden1
                grad_W_h2{j}(i) = grad_W_h2{j}(i) + hidden2_signals(j) * hidden_layer1(i).output;
            end
        end

        % Hidden layer 1
        for j = 1:numHidden1
            grad_b_h1(j) = grad_b_h1(j) + hidden1_signals(j);
            for i = 1:N
                grad_W_h1{j}(i) = grad_W_h1{j}(i) + hidden1_signals(j) * input_layer(i).output;
            end
        end
    end
    %% Weights Update

    % Input -> Hidden1
    for j = 1:numHidden1
        hidden_layer1(j).bias_weight = hidden_layer1(j).bias_weight + eta * grad_b_h1(j) / P;
        for i = 1:N
            hidden_layer1(j).input_connections(i).weight = ...
                hidden_layer1(j).input_connections(i).weight + eta * grad_W_h1{j}(i) / P;
        end
    end

    % Hidden1 -> Hidden2
    for j = 1:numHidden2
        hidden_layer2(j).bias_weight = hidden_layer2(j).bias_weight + eta * grad_b_h2(j) / P;
        for i = 1:numHidden1
            hidden_layer2(j).input_connections(i).weight = ...
                hidden_layer2(j).input_connections(i).weight + eta * grad_W_h2{j}(i) / P;
        end
    end

    % Hidden2 -> Output
    for k = 1:M
        output_layer(k).bias_weight = output_layer(k).bias_weight + eta * grad_b_out(k) / P;
        for j = 1:numHidden2
            output_layer(k).input_connections(j).weight = ...
                output_layer(k).input_connections(j).weight + eta * grad_W_out(k,j) / P;
        end
    end

    % Compute total error over an epoch (1/2 factor included)
    mse = total_error / (P * M) * 2;
    fprintf('Epoch %d | MSE = %.6f\n', epoch, mse);

    % Collected network outputs over an epoch
    err = B - Yhat;
    mse_per_output = mean(err.^2,1);
    disp(mse_per_output);

    % Live Plot
    mse_history(epoch) = mean(mse_per_output);
    set(hLine,'XData',1:epoch,'YData',mse_history(1:epoch));
    drawnow;
end
%%
function hidden_conns = generate_hidden_conns_from(input_units)
    hidden_conns(1, numel(input_units)) = struct('neuron',[],'weight',[]);

    for i=1:numel(input_units)
        hidden_conns(i).neuron = input_units(i);
        hidden_conns(i).weight = randn * 0.1;
    end
end

function output_conns = generate_output_conns_from(hidden_units)
    output_conns(1, numel(hidden_units)) = struct('neuron',[],'weight',[]);

    for i=1:numel(hidden_units)
        output_conns(i).neuron = hidden_units(i);
        output_conns(i).weight = randn * 0.1;
    end
end
