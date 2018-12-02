function [input_od] = relu_backward(output, input, layer)

%% function input
% output.data: output data of relu_forward
% output.diff: gradient w.r.t output.data

%% function output
% input_od: gradient w.r.t input.data

%% here begins the relu forward computation

% initialize
input_od = zeros(size(input.data));

% start to work here to compute input_od

input_od = (input.data > 0).*output.diff;
end
