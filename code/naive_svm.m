if (~ exist("norm_data"))
    load("data.mat");
end
disp("Finished loading data");

trainX = [norm_data{3}; pneu_data{3}];
trainY = [zeros(size(norm_data{3}, 1), 1); ones(size(pneu_data{3}, 1), 1)];

testX = [norm_data{2}; pneu_data{2}];
testY = [zeros(size(norm_data{2}, 1), 1); ones(size(pneu_data{2}, 1), 1)];

mdl = fitcsvm(trainX, trainY);
disp("Finished Training");
predY = predict(mdl, testX);
mean(predY==testY)