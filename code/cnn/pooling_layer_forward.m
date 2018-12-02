function [output] = pooling_layer_forward(input, layer)

%% function input:
% input.batch_size: batch_size of input
% input.height: height of input
% input.width : width of input
% input.data: the actual data of input
% input.data is of size (input.height*input.width*input.channel, input.batch_size)

% layer.k: kernel size of pooling operation
% layers.stride: stride of pooling operation
% layers.pad: pad of pooling operation


%% function output
% output: the output of inner_product_forward

% figure out the output shape
h_in = input.height;
w_in = input.width;
c = input.channel;
batch_size = input.batch_size;
k = layer.k;
layer.pad = 0;
pad = layer.pad;
stride = layer.stride;

h_out = (h_in + 2*pad - k) / stride + 1;
w_out = (w_in + 2*pad - k) / stride + 1;
assert(h_out == floor(h_out), 'h_out is not integer')
assert(w_out == floor(w_out), 'w_out is not integer')
input_n.height = h_in;
input_n.width = w_in;
input_n.channel = c;

% set output shape
output.height = h_out;
output.width = w_out;
output.channel = c;
output.batch_size = batch_size;

% initialize output.data
output.data = zeros(h_out*w_out*c, batch_size);



switch layer.act_type
    case 'MAX'
        % work out the max pooling and compute output.data
        for n = 1:batch_size
            input_n.data = reshape(input.data(:,n), h_in, w_in, c);
            temp_out = zeros(h_out, w_out, c);
            for c_i = 1:c
                for h = 1:h_out
                    for w = 1:w_out
                        h_range = (1+(h-1)*stride):((h-1)*stride+k);
                        w_range = (1+(w-1)*stride):((w-1)*stride+k);
                        A = input_n.data(h_range,w_range,c_i);
                        temp_out(h, w, c_i) = max(max(A));
                    end
                end
            end
            output.data(:,n) = reshape(temp_out,h_out*w_out*c, 1);
        end
    case 'AVE'
        % work out the average pooling and compute output.data
        for n = 1:batch_size
            input_n.data = reshape(input.data(:,n), h_in, w_in, c);
            temp_out = zeros(h_out, w_out, c);
            for c_i = 1:c
                for h = 1:h_out
                    for w = 1:w_out
                        h_range = (1+(h-1)*stride):((h-1)*stride+k);
                        w_range = (1+(w-1)*stride):((w-1)*stride+k);
                        A = input_n.data(h_range,w_range,c_i);
                        temp_out(h, w, c_i) = mean(mean(A));
                    end
                end
            end
            output.data(:,n) = reshape(temp_out,1, h_out*w_out*c);
        end
end
end

