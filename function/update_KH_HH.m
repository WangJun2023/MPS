function [sum_KH, sum_HH] = update_KH_HH(KH, HH, mu, gamma)
[num_sample, ~, num_kernel] = size(KH);
sum_HH = zeros(num_sample, num_sample);
sum_KH = zeros(num_sample, num_sample);
for p = 1 : num_kernel
    sum_HH = sum_HH + mu(p) * HH(:, :, p);
    sum_KH = sum_KH + gamma(p) * KH(:, :, p);
end
end