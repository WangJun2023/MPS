function [F_normalized, S, mu, gamma, obj] = SLgPA(KH, HH, num_cluster, alpha, beta)

[num_sample, ~, num_kernel] = size(KH);
knn = 15;
max_iter = 15;

mu = sqrt(ones(num_kernel, 1) ./ num_kernel);
gamma = sqrt(ones(num_kernel, 1) ./ num_kernel);

for iter = 1 : max_iter
    
    % update sum_HH sum_HK
    [sum_KH, sum_HH] = update_KH_HH(KH, HH, mu, gamma);
    
    % update S
    if iter > 1
        FF = pdist2(F, F);
        
        %         FF = pdist2(D^(-0.5) * F, D^(-0.5) * F);
        
        dist = FF ./ alpha - sum_KH;
        
    else
        dist = - sum_KH;
    end
    
    [S, ~] = update_graph(dist, knn);
    L = diag(sum(S)) - S;
    
    %     D = diag(sum(S) + eps);
    %     L = eye(num_sample) - D^(-0.5) * S * D^(-0.5);
    %
    % update F
    [F, ~] = eigs(2 * L - beta * sum_HH, num_cluster, 'sa');
    
    % update mu
    f_1 = zeros(num_kernel, 1);
    f_2 = zeros(num_kernel, 1);
    for p = 1 : num_kernel
        f_1(p) = trace(F' * HH(:, :, p) * F);
        f_2(p) = trace(KH(:, :, p)' * S);
    end
    mu = f_1 ./ norm(f_1);
    gamma = f_2 ./ norm(f_2);
    
    % update obj
    obj(iter) = trace(F' * L * F);
    
    if iter > 2 && abs( (obj(iter) - obj(iter - 1)) /  obj(iter)) < 1e-5
        break;
    end
    
end

F_normalized = F ./ repmat(sqrt(sum(F .^ 2, 2)), 1, num_cluster);

end