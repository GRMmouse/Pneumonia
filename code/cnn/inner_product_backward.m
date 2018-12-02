function [param_grad, input_od] = inner_product_backward(output, input, layer, param)

%% function input
% output.data: output data of inner_product_forward
% output.diff: the gradient w.r.t otuput.data  500*64

%% function output
% param_grad.w: gradient w.r.t param.w 800*500
% param_grad.b: gradient w.r.t param.b 1*500
% input_od: gradient w.r.t input.data 800*64

%% here begins inner product backward computation

% initialize the gradient w.r.t param
param_grad.w = zeros(size(param.w)); % gradient w.r.t param.w
param_grad.b = zeros(size(param.b)); % gradient w.r.t param.b
input_od = zeros(size(input.data));

% start to work here to compute param_grad.w, param_grad.b, input_od

input_od = param.w * output.diff;
param_grad.w = input.data * output.diff';
param_grad.b = sum(output.diff', 1);

end
