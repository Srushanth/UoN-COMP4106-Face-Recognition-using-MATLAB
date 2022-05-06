% https://codetobuy.com/downloads/matlab-code-for-face-recognition-based-on-histogram-of-oriented-gradients-hog/
% https://cmp.felk.cvut.cz/~spacelib/faces/faces96.html

clc;
clear;
close all;

%% Load Images Information from FaceDatabase Face Database Directory

faceDatabase = imageSet('newDS','recursive');

%% Display Various Sets of First Face

figure(1); montage(faceDatabase(1).ImageLocation);

title('Images of the First Face');

%%  Display Test Image and Database Side-Side

testperson = 1;

galleryImage = read(faceDatabase(testperson),1);

figure(2);

imageList = strings([1 , size(faceDatabase,2)]);

for i=1:size(faceDatabase,2)
    
    imageList(i) = faceDatabase(i).ImageLocation(5);

end

subplot(1,2,1);imshow(galleryImage); title('Test Image')

subplot(1,2,2);montage(imageList); title('Database')

diff = zeros(1,9);

%% Split Database into Training & Test Sets

[training,test] = partition(faceDatabase,[0.8 0.2]);

%% Extract and display Histogram of Oriented Gradient Features for single face

person = 1;

[hogFeature, visualization]= extractHOGFeatures(read(training(person),1));

figure(3);

subplot(1,2,1);imshow(read(training(person),1));title('Input Face');

subplot(1,2,2);plot(visualization);title('HoG Feature');

%% Extract HOG Features for training set

trainingFeatures = zeros(size(training,2)*training(1).Count,39204);

featureCount = 1;
personIndex = strings([1 , size(training, 2)]);
trainingLabel = strings([1 , size(training, 2)*training(i).Count]);

for i=1:size(training,2)
    for j = 1:training(i).Count
        trainingFeatures(featureCount,:) = extractHOGFeatures(read(training(i),j));
        trainingLabel{featureCount} = training(i).Description;
        featureCount = featureCount + 1;
    end
    
    personIndex{i} = training(i).Description;

end

%% Create 40 class classifier using fitcecoc

faceClassifier = fitcecoc(trainingFeatures,trainingLabel);
save faceClassifier;

%% Test Images from Test Set

person = 2;

queryImage = read(test(person),1);

queryFeatures = extractHOGFeatures(queryImage);

personLabel = predict(faceClassifier,queryFeatures);

% Map back to training set to find identity

booleanIndex = strcmp(personLabel, personIndex);

integerIndex = find(booleanIndex);

subplot(1,2,1);imshow(queryImage);title('Query Face');

subplot(1,2,2);imshow(read(training(integerIndex),1));title('Matched Class');

