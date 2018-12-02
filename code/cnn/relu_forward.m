function [output] = relu_forward(input, layer)

%% function input
% input.data: actual input data of relu layer

%% function output
% output: the output of relu_forward function

%% here begins the relu forward computation

% set the shape of output
output.height = input.height;
output.width = input.width;
output.channel = input.channel;
output.batch_size = input.batch_size;

% start to work here to compute output.data
output.data = input.data.*(input.data > 0);

end
