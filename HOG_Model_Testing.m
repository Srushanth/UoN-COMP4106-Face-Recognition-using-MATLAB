
%%
clc;
clear;
close all;

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
