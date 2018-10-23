function [score, latent, var_explain] = choose_pc(x, k)
    [~, S, V] = svd(x, 'econ');
    S = diag(S); %Notice SVD returns result in decending order
    score = V(:,1:k);
    latent = S(1:k);
    var_explain = sum(latent.^2)/sum(S.^2);
end

