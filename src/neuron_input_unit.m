classdef neuron_input_unit < handle

    properties
        output
    end

    methods
        % Constructor
        function obj = neuron_input_unit(input)
                obj.output = input;
        end
    end
end
