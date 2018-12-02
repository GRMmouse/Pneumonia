% Accuracy: .7468
% Precision: .7180
% Recall: .9795

%% Load Data
if (~ exist("norm_data"))
    load("data.mat");
end
disp("Finished loading data");

trainX = [norm_data{3}; pneu_data{3}];
trainY = [zeros(size(norm_data{3}, 1), 1); ones(size(pneu_data{3}, 1), 1)];

testX = [norm_data{2}; pneu_data{2}];
testY = [zeros(size(norm_data{2}, 1), 1); ones(size(pneu_data{2}, 1), 1)];

%% PCA
[score, latent, var_explained] = choose_pc(trainX, 32);
sprintf("%.2f of total variance explained", var_explained)

%% Classification
mdl = fitcsvm(trainX, trainY);
disp("Finished Training");
predY = predict(mdl, testX);

%% Evaluation
tp = sum(testY==1 & predY==1);
fp = sum(testY==0 & predY==1);
tn = sum(testY==0 & predY==0);
fn = sum(testY==1 & predY==0);
acc = (tp+tn)/size(predY, 1)
precision = tp/(tp+fp)
recall = tp/(tp+fn)