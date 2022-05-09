clc;
clear;
close all;

allImages=imageDatastore('newDS', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

[Train ,Test] = splitEachLabel(allImages,0.8,'randomized');
fc = fullyConnectedLayer(4);
net = alexnet;
ly = net.Layers;
ly(23) = fc;
cl = classificationLayer;
ly(25) = cl; 
% options for training the net if your newnet performance is low decrease
% the learning_rate
learning_rate = 0.00001;
opts = trainingOptions("rmsprop","InitialLearnRate",learning_rate,'MaxEpochs',5,'MiniBatchSize',64,'Plots','training-progress');
[newnet,info] = trainNetwork(Train, ly, opts);
[predict,scores] = classify(newnet,Test);
names = Test.Labels;
pred = (predict==names);
s = size(pred);
acc = sum(pred)/s(1);
fprintf('The accuracy of the test set is %f %% \n',acc*100);