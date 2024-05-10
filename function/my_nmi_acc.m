function [res] = my_nmi_acc(F_normalized, gt, num_cluster)
stream = RandStream.getGlobalStream;
reset(stream);
max_rep = 20;
res_rep = zeros(max_rep, 8);

F_normalized = F_normalized ./ repmat(sqrt(sum(F_normalized .^ 2, 2)), 1, size(F_normalized, 2));

for rep = 1 : max_rep
    
    pre = kmeans(F_normalized, num_cluster, 'maxiter', 100, 'replicates', 20, 'emptyaction', 'singleton');
   
%     pre = litekmeans(F_normalized, num_cluster, 'MaxIter', 100, 'Replicates', 20);

    res_rep(rep,:) = Clustering8Measure(gt, pre);
    
end

res = [mean(res_rep); std(res_rep)];

end