function [input_od] = pooling_layer_backward(output, input, layer)

%% function input:
% input: input of pooling_layer_forward
% output: output of pooling_layer_forward

% layer.k: kernel size of pooling operation
% layer.stride: stride of pooling operation
% layer.pad: pad of pooling operation


%% function output
% input_od: gradient w.r.t input.data

% initialize
input_od = zeros(size(input.data));
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
input_temp = zeros(h_in*w_in*c, batch_size);

switch layer.act_type
    case 'MAX'
        % work out the max pooling backward and compute input_od
        for n = 1:batch_size
            input_n.data = reshape(input.data(:,n), h_in, w_in, c);
            output_n.diff = reshape(output.diff(:,n), h_out, w_out, c);
            temp = zeros(h_in, w_in, c);
            for c_i = 1:c
                for h = 1:h_out
                    for w = 1:w_out
                        h_range = (1+(h-1)*stride):((h-1)*stride+k);
                        w_range = (1+(w-1)*stride):((w-1)*stride+k);
                        A = input_n.data(h_range,w_range,c_i);
                        [B, max_rows] = max(A);
                        [C, max_col] = max(B);
                        max_row = max_rows(max_col)+(h-1)*stride;
                        max_col = max_col+(w-1)*stride;
                        temp(max_row, max_col, c_i) = output_n.diff(h, w, c_i);
                    end
                end
            end
            input_temp(:,n) = reshape(temp, h_in*w_in*c, 1);
        end

        input_od = reshape(input_temp, size(input_od));
    case 'AVE'
        % work out the max pooling backward and compute input_od
        for n = 1:batch_size
            input_n.data = reshape(input.data(:,n), h_in, w_in, c);
            output_n.diff = reshape(output.diff(:,n), h_out, w_out, c);
            temp = zeros(h_in, w_in, c);
            for c_i = 1:c
                for h = 1:h_out
                    for w = 1:w_out
                        h_range = (1+(h-1)*stride):((h-1)*stride+k);
                        w_range = (1+(w-1)*stride):((w-1)*stride+k);
                        A = output_n.diff(h, w, c_i)/k^2*ones(k, k);
                        temp(h_range,w_range,c_i) = A;
                    end
                end
            end
            input_temp(:,n) = reshape(temp, h_in*w_in*c, 1);
        end

        input_od = reshape(input_temp, size(input_od));
end

end