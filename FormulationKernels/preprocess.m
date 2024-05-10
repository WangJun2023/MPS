function [KH, HP, num_kernel] = preprocess(fea, num_cluster, dim_c)
num_view = length(fea);
num_sample = size(fea{1}, 1);

% normalize data
fea_normalized = cell(num_view, 1);
for v = 1 : num_view
    fea_normalized{v} = normalize_fea(fea{v});
end

% construct kernels
num_kernel = num_view * 5;
KH = zeros(num_sample, num_sample, num_kernel);
for v = 1 : num_view
    options.KernelType = 'Gaussian';
    KH( : , : , 1 + (v - 1) * 5) = construct_kernel(fea_normalized{v}, [], options);
    options.KernelType = 'Polynomial';
    options.d = 3;
    KH( : , : , 2 + (v - 1) * 5) = construct_kernel(fea_normalized{v}, [], options);
    options.KernelType = 'Linear';
    KH( : , : , 3 + (v - 1) * 5) = construct_kernel(fea_normalized{v}, [], options);
    options.KernelType = 'Sigmoid';
    options.c = 0;
    options.d = 0.1;
    KH( : , : , 4 + (v - 1) * 5) = construct_kernel(fea_normalized{v}, [], options);
    options.KernelType = 'InvPloyPlus';
    options.c = 0.01;
    options.d = 1;
    KH( : , : , 5 + (v - 1) * 5) = construct_kernel(fea_normalized{v}, [], options);
end

% normalize kernels
KH = knorm(kcenter(KH));


for v = 1 : num_kernel
    KH(:,:,v) = (KH(:,:,v)+KH(:,:,v)')/2;
    for d = 1 : dim_c
        k = d * num_cluster;
        [Hp{d}, ~] = eigs(KH(:,:,v), k, 'la');    
    end
    HP{v} = Hp;
end




end
