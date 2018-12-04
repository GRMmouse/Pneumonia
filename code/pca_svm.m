% Accuracy: .7676
% Precision: .7333
% Recall: .9872

%% Parameters
num_pcs = 50;

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
[score, latent, var_explained] = choose_pc(trainX, num_pcs);
fprintf("%.2f%% of total variance explained\n", var_explained*100);
trainX = trainX*score;
testX = testX*score;

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

%% Plot
close all
[~, inds] = sort(abs(mdl.Beta), 'descend')
figure
scatter3(testX(testY==1 & predY==1, inds(1)), testX(testY==1 & predY==1, inds(2)),...
    testX(testY==1 & predY==1, inds(3)), 'c.')
hold on
scatter3(testX(testY==0 & predY==1, inds(1)), testX(testY==0 & predY==1, inds(2)),...
    testX(testY==0 & predY==1, inds(3)), 'mx')
scatter3(testX(testY==0 & predY==0, inds(1)), testX(testY==0 & predY==0, inds(2)),...
    testX(testY==0 & predY==0, inds(3)), 'y.')
scatter3(testX(testY==1 & predY==0, inds(1)), testX(testY==1 & predY==0, inds(2)),...
    testX(testY==1 & predY==0, inds(3)), 'kx')
legend('tp','fp','tn','fn');
xlabel('PC 1');
ylabel('PC 2');
zlabel('PC 3');
axis square
rotate3d on