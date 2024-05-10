function  [F, obj] = main(KH, HP, num_kernel, dim_c, num_cluster, alpha, sel_kernel)

%% Initialization
beta = sqrt(ones(1, dim_c)/dim_c);
mu1 = sqrt(ones(num_kernel, 1)/num_kernel);
mu2 = sqrt(ones(num_kernel, 1)/num_kernel);

[num_sample, ~, num_kernel] = size(KH);

for v = 1 : num_kernel
    sum_KH = KH(:, :, v);
    [S{v}, ~] = update_graph(-sum_KH, 15);
end

maxIter = 50;

flag = 1;
iter = 0;


while flag
    iter = iter + 1;      
    
    %% Update F
    tmp = zeros(num_sample);
    for v = 1 : num_kernel
        Hp = HP{v};
        for d = 1 : dim_c
            tmp = tmp + (1-alpha) * mu1(v)*beta(d) * Hp{d}*Hp{d}';
        end
        tmp = tmp + alpha * mu2(v) * S{v};
    end
    [F,~] = eigs(tmp, num_cluster, 'la');

    %% Update gamma
    f_1 = zeros(num_kernel, 1);
    for v = 1 : num_kernel
        tmp = zeros(num_sample);
        Hp = HP{v};
        for d = 1 : dim_c
            tmp = tmp + beta(d)*Hp{d}*Hp{d}';
        end
        f_1(v) = trace(alpha*F'*S{v}*F+(1-alpha)*F'*tmp*F);     
    end
    dis = f_1 ./ norm(f_1);
    [~, gamma] = selec_max(dis, num_kernel, sel_kernel);

    %% Update feature-weight
    f_2 = zeros(num_kernel, 1);
    for v = 1 : num_kernel
        tmp = zeros(num_sample);
        Hp = HP{v};
        for d = 1 : dim_c
            tmp = tmp + gamma(v)*beta(d)*Hp{d}*Hp{d}';
        end
        f_2(v) = trace((1-alpha)*F'*tmp*F);     
    end
    mu1 = f_2 ./ norm(f_2);
    
    %% Update similarity graph weight
    f_3 = zeros(num_kernel, 1);
    for v = 1 : num_kernel
        f_3(v) = trace(alpha*F'*gamma(v)*S{v}*F);     
    end
    mu2 = f_3 ./ norm(f_3);

    %% Update beta
    f_4 = zeros(1, dim_c);
    for d = 1 : dim_c
        tmp = zeros(num_sample);
        for v = 1 : num_kernel
            Hs = HP{v}{d};
            tmp = tmp + mu1(v)*Hs*Hs';
        end
        f_4(d) = trace((1-alpha)*F'*tmp*F);
    end
    beta = f_4 ./ norm(f_4);

    %% Cal obj
%     obj1 = 0;
%     obj2 = 0;
%     for v = 1 : num_kernel
%         obj1 = obj1 + trace(mu2(v)*F'*S{v}*F);
%         Hp = HP{v};
%         for d = 1 : dim_c
%             obj2 = obj2 + trace(mu1(v)*beta(d)*F'*Hp{d}*Hp{d}'*F);
%         end
%     end
%     obj(iter) = alpha*obj1 + (1-alpha)*obj2;

    %% Cal obj
    obj1 = 0;
    obj2 = 0;
    for v = 1 : num_kernel
        obj1 = obj1 + norm(F*F'-mu2(v)*S{v}, 'fro');
        Hp = HP{v};
        for d = 1 : dim_c
            obj2 = obj2 + norm(F*F'-mu1(v)*beta(d)*Hp{d}*Hp{d}', 'fro');
        end
    end
    obj(iter) = alpha*obj1 + (1-alpha)*obj2;


    if (iter>2) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-6 || iter>maxIter)
        flag =0;
    end
end
end

function [vec, idx] = selec_max(alpha, num_kernel, k)
vec = zeros(num_kernel, 1);
idx = zeros(num_kernel, 1);
for i = 1 : k
    col = find(alpha==max(alpha));
    vec(col) = alpha(col);
    alpha(col) = 0;
    idx(col) = 1;
end

end
