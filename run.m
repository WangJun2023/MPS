close all;
clear;
clc;
warning off;


dataPath = 'Datasets/Multi_view-Datasets/';
datasetName = {'MSRC_v1', 'BBCSport', 'BRCA','Caltech101-20', 'EMNIST',...
    '3sources', 'ORL', 'scene-15_3v'};


ResSavePath = 'MPS/Res/';
MaxResSavePath = 'MPS/maxRes/';

if(~exist(ResSavePath,'file'))
    mkdir(ResSavePath);
    addpath(genpath(ResSavePath));
end

if(~exist(MaxResSavePath,'file'))
    mkdir(MaxResSavePath);
    addpath(genpath(MaxResSavePath));
end

for dataIndex = 1
    dataName = [dataPath datasetName{dataIndex} '.mat'];
    load(dataName, 'fea', 'gt');

    ResBest = zeros(1, 8);
    ResStd = zeros(1, 8);

    % Data Preparation
    tic;
    num_cluster = length(unique(gt));

    dim_c = 5;

    [KH, HP, num_kernel] = preprocess(fea, num_cluster, dim_c);
    time1 = toc;

    % parameters setting
    r1 = [0.1 : 0.1 : 0.9];
    r2 = [0.1 : 0.1 : 0.9] * num_kernel;

    acc = zeros(length(r1), length(r2));
    nmi = zeros(length(r1), length(r2));
    ari = zeros(length(r1), length(r2));
    Fscore = zeros(length(r1), length(r2));

    idx = 1;
    for r1Index = 1:length(r1)
        r1Temp = r1(r1Index);
        for r2Index = 1:length(r2)
            r2Temp = ceil(r2(r2Index));
            tic;
            % Main algorithm
            fprintf('Please wait a few minutes\n');
            disp(['Dataset: ', datasetName{dataIndex}, ...
                ', --r1--: ', num2str(r1Temp), ', --r2--: ', num2str(r2Temp)]);

            [F, obj] = main(KH, HP, num_kernel, dim_c, num_cluster, r1Temp, r2Temp);

            time2 = toc;

            tic;
            [res] = my_nmi_acc(real(F), gt, num_cluster);
            time3 = toc;

            Runtime(idx) = time1 + time2 + time3/20;
            disp(['runtime: ', num2str(Runtime(idx))]);
            idx = idx + 1;
            tempResBest(1, :) = res(1, :);
            tempResStd(1, :) = res(2, :);

            acc(r1Index, r2Index) = tempResBest(1, 7);
            nmi(r1Index, r2Index) = tempResBest(1, 4);
            ari(r1Index, r2Index) = tempResBest(1, 5);
            Fscore(r1Index, r2Index) = tempResBest(1, 1);

            resFile = [ResSavePath datasetName{dataIndex}, '-ACC=', num2str(tempResBest(1, 7)), ...
                '-r1=', num2str(r1Temp), '-r2=', num2str(r2Temp), '.mat'];
            save(resFile, 'tempResBest', 'tempResStd');

            if tempResBest(1, 7) > ResBest(1, 7)
                ResBest(1, :) = tempResBest(1, :);
                ResStd(1, :) = tempResStd(1, :);
            end
        end
    end
    aRuntime = mean(Runtime);
    resFile2 = [MaxResSavePath datasetName{dataIndex}, '-ACC=', num2str(ResBest(1, 7)), '.mat'];
    save(resFile2, 'ResBest', 'ResStd', 'acc', 'nmi', 'ari', 'Fscore', 'aRuntime');
end

