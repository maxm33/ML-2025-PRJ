classdef neuron_hidden_unit < handle

    properties
        bias_weight = randn * 0.1
        input_connections               % struct: (unit, weight)
        net
        output
    end

    methods
        % Constructor
        function obj = neuron_hidden_unit(input_connections)
            obj.input_connections = input_connections;
        end

        % Leaky ReLU activation function
        function y = Leaky_ReLU(~, x)
            y = max(0.01 * x, x);       % slope 0.01 for negative x
        end

        % Function derivative
        function dy = Leaky_ReLU_derivative(~, x)
            dy = 0.01 * (x <= 0) + 1 * (x > 0);
        end

        % Feedforward computation
        function compute(this)
            this.net = this.bias_weight;

            for i = 1:length(this.input_connections)
                unit_i = this.input_connections(i).neuron;
                weight_ji = this.input_connections(i).weight;

                this.net = this.net + weight_ji * unit_i.output;
            end

            this.output = this.Leaky_ReLU(this.net);
        end
    end
end
