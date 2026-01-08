classdef neuron_hidden_unit_tahn < handle
    properties
        bias_weight = 0
        input_connections
        net
        output
    end
    methods
        function obj = neuron_hidden_unit_tahn(input_connections)
            obj.input_connections = input_connections;
        end
        
        % Attivazione Tanh
        function y = activation(~, x)
            y = tanh(x);
        end
        
        % Derivata della Tanh: 1 - tanh(x)^2
        function dy = activation_derivative(~, x)
            dy = 1 - tanh(x)^2;
        end
        
        function compute(this)
            this.net = this.bias_weight;
            for i = 1:length(this.input_connections)
                this.net = this.net + this.input_connections(i).weight * this.input_connections(i).neuron.output;
            end
            this.output = this.activation(this.net);
        end
    end
end