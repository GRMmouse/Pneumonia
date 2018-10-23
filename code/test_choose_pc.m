%% Initialization
x = data;
x_test = x(7,:); % Specific Images
h_out = 224;
w_out = 224;

%% Plot
figure
subplot(2,2,1)
imshow(reshape(x_test, h_out, w_out));
title('Original Image');
for id = 2:4
    subplot(2,2,id);
    k = 2*id-3;
    [score, latent, var_explain] = choose_pc(x, k);
    x_reco = x_test*score*score';
    imshow(reshape(x_reco, h_out, w_out));
    title(sprintf('k = %d (%.2f %%)', k, var_explain*100))
end