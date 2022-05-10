clear all;
close all;
trainPath='.\FaceDatabase\Train\'; % These training/testing folders need to be in the same root folder of this code. 
testPath='.\FaceDatabase\Test\';   % Or you can use the full folder path here
%% Retrive training and testing images

[trainImgSet, trainPersonID]=loadTrainingSet(trainPath); % load training images

size(trainImgSet)  % if successfully loaded this should be with dimension of 600,600,3,100

%% Now we need to do facial recognition: Baseline Method
tic;
   outputID=FaceRecognition(trainImgSet, trainPersonID, testPath);
runTime=toc

load testLabel
correctP=0;
for i=1:size(testLabel,1)
    if strcmp(outputID(i,:),testLabel(i,:))
        correctP=correctP+1;
    end
end
recAccuracy=correctP/size(testLabel,1)*100  %Recognition accuracy

%% Method developed by you
%% Load Image Dataset
raw_data = imageSet('newDS', 'recursive');

%% Load the model
load classifier.mat

%% Testing the model
total_test_samples = 0;
total_matched_samples = 0;

% Iterating through all the images
for i=1:size(raw_data, 2)
    for j = 1:raw_data(i).Count
        total_test_samples = total_test_samples + 1;

        % Reading the image
        test_image = read(raw_data(i), j);

        % Extracting HoG Features
        extracted_features = extractHOGFeatures(test_image);

        % Predicting the image label
        predicted_label = predict(classifier, extracted_features);
        if raw_data(i).Description == string(predicted_label)
            total_matched_samples = total_matched_samples + 1;
        end
    end
end

%% Accuracy of the model
model_accuracy = (total_matched_samples/total_test_samples) * 100;
msgbox(strcat('Model accuracy = ', string(model_accuracy), '%'));


