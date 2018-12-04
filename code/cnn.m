% Accuracy: 
% Precision: 
% Recall: 


%% Load Data
if (~ exist("norm_data", "var"))
    load("data.mat");
end
disp("Finished loading data");

trainX = [norm_data{3}; pneu_data{3}]';
trainY = [zeros(size(norm_data{3}, 1), 1); ones(size(pneu_data{3}, 1), 1)];

testX = [norm_data{2}; pneu_data{2}]';
testY = [zeros(size(norm_data{2}, 1), 1); ones(size(pneu_data{2}, 1), 1)];

validX = [norm_data{1}; pneu_data{1}]';
validY = [zeros(size(norm_data{1}, 1), 1); ones(size(pneu_data{1}, 1), 1)];

%% Classification
% Network Configuration
layers{1}.type = 'DATA';
layers{1}.height = 32;
layers{1}.width = 32;
layers{1}.channel = 1;
layers{1}.batch_size = 64;

layers{2}.type = 'CONV'; % second layer is conv layer
layers{2}.num = 10; % number of output channel
layers{2}.k = 3; % kernel size
layers{2}.stride = 1; % stride size
layers{2}.pad = 0; % padding size
layers{2}.group = 1; % group of input feature maps
                     % you can ignore this 

                     
layers{3}.type = 'POOLING'; % third layer is pooling layer
layers{3}.act_type = 'MAX'; % use max pooling
layers{3}.k = 2; % kernel size
layers{3}.stride = 2; % stride size
layers{3}.pad = 0; % padding size

layers{4}.type = 'CONV';
layers{4}.k = 2;
layers{4}.stride = 1;
layers{4}.pad = 0;
layers{4}.group = 1;
layers{4}.num = 20;

layers{5}.type = 'POOLING';
layers{5}.act_type = 'MAX';
layers{5}.k = 2;
layers{5}.stride = 2;
layers{5}.pad = 0;

layers{6}.type = 'IP'; % inner product layer
layers{6}.num = 50; % number of output dimension
layers{6}.init_type = 'uniform'; % initialization method 

layers{7}.type = 'RELU'; % relu layer

layers{8}.type = 'LOSS'; % loss layer
layers{8}.num = 1; % number of classes


m_train = size(trainX, 2);
batch_size = 64;

% learning rate parameters
mu = 0.9; % momentum
epsilon = 0.01; % initial learning rate
gamma = 0.0001; 
power = 0.75;
weight_decay = 0.0005; % weight decay on w


% display information
test_interval = 500;
display_interval = 5;
snapshot = 5000;
max_iter = 1000;

% initialize all parameters in each layers
params = init_convnet(layers);

% buffer for sgd momentum
param_winc = params;
for l_idx = 1:length(layers)-1
    param_winc{l_idx}.w = zeros(size(param_winc{l_idx}.w));
    param_winc{l_idx}.b = zeros(size(param_winc{l_idx}.b));
end

for iter = 1 : max_iter
    % randomly fetch a batch
    id = randi([1 m_train], batch_size, 1);

    % forward and backward
    [cp, param_grad] = conv_net(params, layers, trainX(:, id), trainY(id));

    % get learning rate
    rate = get_lr(iter, epsilon, gamma, power);
    % update param with sgd momentum
    [params, param_winc] = sgd_momentum(rate, mu, weight_decay, params, param_winc, param_grad);

    if mod(iter, display_interval) == 0
        fprintf('iteration %d training cost = %f accuracy = %f\n', iter, cp.cost, cp.percent);
    end
    
    % validate on the test partition
    if mod(iter, test_interval) == 0
        layers{1}.batch_size = 10000;
        [cptest] = conv_net(params, layers, testX, testY);
        layers{1}.batch_size = 64;
        fprintf('test accuracy: %f \n\n', cptest.percent);
    end
    % save model
    if mod(iter, snapshot) == 0
        filename = 'net.mat';
        save(filename, 'params');
    end
end
