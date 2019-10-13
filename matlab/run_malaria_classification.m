%% Malaria Classification
% Author: HONGYU WU
% Date: 04/23/2019


%% Read and Visualize Different Categories

% Clear Workspace
clear all; close all;clc;

%% Image Datastore
imds = imageDatastore('E:\cell_images\',...
     'IncludeSubFolders', 1, 'Labelsource','foldernames');

%% Dataset count
% Count of Each Label
total_count=countEachLabel(imds);
disp(total_count);

% Double of Labels
labels=double(imds. Labels);
figure;
bar(total_count.Count);
title('Distribution of Dataset');
xlabel('Class #');
ylabel('Count');

%% Visualize Each Category
% orginal image
for idx=1:length(unique(labels))
    
    % Determine each category Index
    cat_idx=find(labels==idx,1,'first');
    
    % Read the Image
    I=readimage(imds,cat_idx);
    
    
    % Visualize the image
    figure;    
    imagesc(I);
    hold on;
    title(sprintf('Class %d',idx));
    size(I)
    pause;
    
end
imds.ReadFcn=@(filename)preprocess_image_malaria(filename);
% image after preprocessing
for idx=1:length(unique(labels))
    
    % Determine each category Index
    cat_idx=find(labels==idx,1,'first');
    
    % Read the Image
    I=readimage(imds,cat_idx);
        
    % Visualize the image
    figure;
    imagesc(I);
    hold on;
    title(sprintf('Class %d',idx));
    size(I)
    pause;    
end
%% Train and Validation Datasets

% Split Dataset - Training and Validation

train_test_percentage = 0.8;
[imdsTrain, imdsTest] = splitEachLabel(imds, train_test_percentage);
train_valid_percentage = 0.9
[imdsTrain, imdsValid] = splitEachLabel(imdsTrain, train_valid_percentage);

% Count Labels
test_count=countEachLabel(imdsTest);
train_count=countEachLabel(imdsTrain);
valid_count=countEachLabel(imdsValid);

figure;
bar([train_count.Count,valid_count.Count,test_count.Count])
legend('Train Dataset','Validation Dataset','Test Dataset');
xlabel('Class #');
ylabel('Count');

%% CNN
img_size=[32 32 3];

layers=[
    % Input Image Layer
    imageInputLayer(img_size);
    convolution2dLayer(3,64);
    batchNormalizationLayer;
    reluLayer;
    maxPooling2dLayer(2,'Stride',2);
    
    
    convolution2dLayer(3,128);
    batchNormalizationLayer;
    reluLayer;   
    maxPooling2dLayer(2,'Stride',2);
 
    convolution2dLayer(3,256);
    batchNormalizationLayer;
    reluLayer;   
    maxPooling2dLayer(2,'Stride',2);
    
    fullyConnectedLayer(192);
    dropoutLayer(0.5);
    fullyConnectedLayer(length(unique(labels)));
    
    softmaxLayer;
    classificationLayer;
    ];

% use analyzenetwork(layers) to check architecture in Matlab2018B or later
analyzeNetwork(layers)

% Training Options
options = trainingOptions('adam', 'MiniBatchSize', 32, 'MaxEpochs', 6,...
     'ValidationData', imdsValid, 'ValidationFrequency', 100,...
     'plots', 'training-progress','Verbose',1,'ValidationPatience', 5)
 
% Data Augmentation
imageAugumenter=imageDataAugmenter('RandRotation',[-10 10],'RandXTranslation',[-3 3],'RandYTranslation',[-3 3]);
augimds = augmentedImageDatastore(img_size,imdsTrain,'DataAugmentation',imageAugumenter);

% Train the Network
net = trainNetwork(augimds,layers,options);
% net = trainNetwork(imdsTrain,layers,options);

% Test the Network
[YPred,scores] = classify(net,imdsTest);
YTest = imdsTest.Labels;
accuracy = sum(YPred == YTest)/numel(YTest)
plotconfusion(YPred',imdsTest.Labels'); 

%ROC
figure;
[X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(double(YTest),scores(:,1),1);
plot(X,Y);
grid
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for Classification CNN')