function [S_i, Q] = update_graph(Dist, knn)
[num_sample, ~] = size(Dist);
[Dist_sorted, idx] = sort(Dist, 2);
S_i = zeros(num_sample, num_sample);
weight = zeros(num_sample, 1);
for i = 1 : num_sample
    di = Dist_sorted(i, 2 : knn + 2);
    weight(i) = 0.5 * (knn * di(knn + 1) - sum(di(1 : knn)));
    id = idx(i, 2 : knn + 2);
    S_i(i, id) = (di(knn + 1) - di) / (knn * di(knn + 1) - sum(di(1 : knn)) + eps);
end
Q = mean(weight);
S_i = 0.5 * (S_i + S_i');
end