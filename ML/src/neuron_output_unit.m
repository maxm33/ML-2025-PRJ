classdef neuron_output_unit < handle

    properties
        bias_weight = 0
        input_connections               % struct: (unit, weight)
        net
        output
    end

    methods
        % Constructor
        function obj = neuron_output_unit(input_connections)
            obj.input_connections = input_connections;
        end

        % Output computation
        function compute(this)
            this.net = this.bias_weight;

            for i = 1:length(this.input_connections)
                unit_i = this.input_connections(i).neuron;
                weight_ji = this.input_connections(i).weight;

                this.net = this.net + weight_ji * unit_i.output;
            end

            this.output = this.net;     % Linear outputs
        end
    end
end
