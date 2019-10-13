%% Malaria Classification with transfer learning(Alexnet)
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
%% image preprocessing
imds.ReadFcn=@(filename)preprocess_image_malaria_alexnet(filename);

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
train_valid_percentage = 0.9;
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

%% Load Pretrained Network

net = alexnet;
analyzeNetwork(net)
inputSize = net.Layers(1).InputSize

%% Replace Final Layers

% Extract all layers, except the last three, from the pretrained network.
layersTransfer = net.Layers(1:end-3)
numClasses = length(unique(labels))

% CNN

% To learn faster in the new layers than in the transferred layers, increase the 
% WeightLearnRateFactor and BiasLearnRateFactor values of the fully connected layer
img_size=[227 227 3];
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer
    ]

analyzeNetwork(layers)

%% Train Network

% Training Options
options = trainingOptions('adam', 'MiniBatchSize', 32, 'MaxEpochs', 6,...
     'InitialLearnRate',1e-4, ...
     'ValidationData', imdsValid, 'ValidationFrequency', 100,...
     'plots', 'training-progress','Verbose',1,'ValidationPatience', 5)
 
% Data Augumentation
imageAugumenter=imageDataAugmenter('RandRotation',[-10 10],'RandXTranslation',[-3 3],'RandYTranslation',[-3 3]);
augimds = augmentedImageDatastore(img_size,imdsTrain,'DataAugmentation',imageAugumenter);

% Train the Network
netTransfer = trainNetwork(augimds,layers,options);
% net = trainNetwork(imdsTrain,layers,options);

% Test the Network
[YPred,scores] = classify(netTransfer,imdsTest);
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
