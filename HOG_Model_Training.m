% https://codetobuy.com/downloads/matlab-code-for-face-recognition-based-on-histogram-of-oriented-gradients-hog/
% https://cmp.felk.cvut.cz/~spacelib/faces/faces96.html
%%
clc;
clear;
close all;

%% Load Image Dataset
raw_data = imageSet('newDS', 'recursive');

%% Check if the loaded data is valid
testing_image = read(raw_data(1), 1);

list_of_images = strings([1 , size(raw_data, 2)]);

% Loop to access every directory
for i=1:size(raw_data, 2)
    % Collecting the 1st image from the directory
    list_of_images(i) = raw_data(i).ImageLocation(1);
end

% Display the sample images
figure('Name', 'Test Image');
subplot(1,2,1);
imshow(testing_image); title('Test Image');
subplot(1, 2, 2);
montage(list_of_images); title('Random Test Images');
pause;

%% Split the training & testing data to 70% and 30% resp.
[train, test] = partition(raw_data, [0.7 0.3]);

%% Check if the splitted data is valid
train_image = read(train(1), 1);
test_image = read(test(2), 1);

% Display the sample images
figure('Name', 'Test, Train & Validation Images');
subplot(1,2,1);
montage(train_image); title('Test Images');

subplot(1, 2, 2);
montage(test_image); title('Train Images');
pause;

%% Extracting the HOG (Histogram of Oriented Gradients) Features from test image
test_image = read(train(1), 1);
[hog_features, visualization]= extractHOGFeatures(test_image);

figure('Name', 'HoG Features');
subplot(1,2,1);
imshow(test_image);title('Input Face');

subplot(1,2,2);
plot(visualization);title('HoG Feature');
pause;

%% Extracting the HOG (Histogram of Oriented Gradients) Features from train set
train_features = zeros(size(train, 2) * train(1).Count, 39204);
train_labels = strings([1 , size(train, 2) * train(i).Count]);
unique_labels = strings([1 , size(train, 2)]);
total_features = 1;

for i=1:size(train,2)
    for j = 1:train(i).Count
        train_features(total_features,:) = extractHOGFeatures(read(train(i),j));
        train_labels{total_features} = train(i).Description;
        total_features = total_features + 1;
    end
    unique_labels{i} = train(i).Description;
end

%% Create and save classifier
classifier = fitcecoc(train_features, train_labels);
save classifier;
msgbox("Saved the classifier model.");

%% Testing the model
total_test_samples = 0;
total_matched_samples = 0;

% Iterating through all the images
for i=1:size(test, 2)
    for j = 1:test(i).Count
        total_test_samples = total_test_samples + 1;

        % Reading the image
        test_image = read(test(i), j);

        % Extracting HoG Features
        extracted_features = extractHOGFeatures(test_image);

        % Predicting the image label
        predicted_label = predict(classifier, extracted_features);
        if test(i).Description == string(predicted_label)
            total_matched_samples = total_matched_samples + 1;
        end
    end
end

%% Accuracy of the model
model_accuracy = (total_matched_samples/total_test_samples) * 100;
msgbox(strcat('Model accuracy = ', string(model_accuracy), '%'));