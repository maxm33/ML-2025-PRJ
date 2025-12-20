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
numHidden = 50;                             % # of units inside Hidden Layer

eta = 1e-2;                                 % Learning Rate

lambda = 1e-2;                              % factor for L1 Regularization

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

epoch_times = zeros(1, epochs);
rmse_history = zeros(1, epochs);

% Plot Initialization
figure;
hLine = plot(NaN, NaN, 'b-', 'LineWidth', 2);
xlabel('Epoch'); ylabel('RMSE'); title('Learning Curve'); grid on; hold on;

for epoch = 1:epochs
    epoch_time_start = posixtime(datetime('now'));

    total_error = 0;
    Yhat = zeros(P, M);

    % Loop over all patterns 
    for p = 1:P
        %% Feedforward phase
        for i = 1:N
            input_layer(i).output = A(p,i); % load pattern p
        end
        for i = 1:numHidden
            hidden_layer(i).compute();
        end
        outputs = zeros(1,M);
        for i = 1:M
            output_layer(i).compute();
            outputs(i) = output_layer(i).output;
            Yhat(p, :) = outputs;
        end

        denorm_diff = (B(p,:) - outputs) .* std(outputs_TR,0,1);
        total_error = total_error + sqrt(sum(denorm_diff.^2));

        %% Output signals
        output_signals = zeros(1, M);
        for k = 1:M
            output_signals(k) = (B(p,k) - outputs(k));
        end
    
        %% Hidden signals
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
    
        %% Update kj weights (Hidden -> Output)
        for k = 1:M
            output_layer(k).bias_weight = ...
                output_layer(k).bias_weight + eta * output_signals(k);
    
            for j = 1:numHidden
                output_layer(k).input_connections(j).weight = ...
                    output_layer(k).input_connections(j).weight + ...
                    eta * output_signals(k) * hidden_layer(j).output - ...
                    lambda * sign(output_layer(k).input_connections(j).weight);
            end
        end
    
        %% Update ji weights (Input -> Hidden)
        for j = 1:numHidden
            hidden_layer(j).bias_weight = ...
                hidden_layer(j).bias_weight + eta * hidden_signals(j);
    
            for i = 1:numel(hidden_layer(j).input_connections)
                hidden_layer(j).input_connections(i).weight = ...
                    hidden_layer(j).input_connections(i).weight + ...
                    eta * hidden_signals(j) * ...
                    hidden_layer(j).input_connections(i).neuron.output - ...
                    lambda * sign(hidden_layer(j).input_connections(i).weight);
            end
        end
    end

    % RMSE with collected network outputs over an epoch
    err = B - Yhat;
    rmse_per_output = sqrt(mean(err.^2, 1));

    % Compute Mean Euclidian Error over an epoch
    mee = total_error / P;
    rmse_history(epoch) = mean(rmse_per_output);
    fprintf('Epoch %d | RMSE (norm) = %.6f | MEE (og scale) = %.6f\n', epoch, rmse_history(epoch), mee);
    disp(rmse_per_output);

    % Live Plot
    set(hLine,'XData',1:epoch,'YData',rmse_history(1:epoch));
    drawnow;

    % Shuffling the patterns
    perm = randperm(size(A,1));         % random rows order
    A = A(perm, :);
    B = B(perm, :);

    epoch_times(epoch) = posixtime(datetime('now')) - epoch_time_start;
end

fprintf('Total Training Time (seconds) = %.3f | Average Epoch Time (seconds) = %.3f\n', sum(epoch_times), mean(epoch_times));
%%
function hidden_conns = generate_hidden_conns_from(input_units)
    hidden_conns(1, numel(input_units)) = struct('neuron',[],'weight',[]);

    for i = 1:numel(input_units)
        hidden_conns(i).neuron = input_units(i);
        hidden_conns(i).weight = randn * 0.1;
    end
end

function output_conns = generate_output_conns_from(hidden_units)
    output_conns(1, numel(hidden_units)) = struct('neuron',[],'weight',[]);

    for i = 1:numel(hidden_units)
        output_conns(i).neuron = hidden_units(i);
        output_conns(i).weight = randn * 0.1;
    end
end
