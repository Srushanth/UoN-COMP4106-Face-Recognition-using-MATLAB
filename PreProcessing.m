clc;
clear;
close all;

%% Constants
MASTER_DATASET_F_PATH = '.\DS\';
PROCESSED_DATASET_F_PATH = '.\newDS\';
IMAGE_FILE_TYPE = '.jpg';
IMAGE_SHAPE = [277, 277];

%% Gathering the details of the path
files = dir(MASTER_DATASET_F_PATH);

% Get a logical vector that tells which is a directory.
dir_flags = [files.isdir];

% Extract only those that are directories.
sub_folders = files(dir_flags);

% Get only the folder names into a cell array.
subFolderNames = {sub_folders(3:end).name};

%% Checking if the directory exists
if not(isfolder(PROCESSED_DATASET_F_PATH))
    % Creating the new directory
    mkdir(PROCESSED_DATASET_F_PATH);
end

%% Looping through the folders
for j = 1 : length(subFolderNames)

    %% Getting the folder name
    folder_name = subFolderNames(j);

    % Checking if the directory exists
    if not(isfolder(strcat(PROCESSED_DATASET_F_PATH, folder_name)))
        % Creating the new directory
        mkdir(PROCESSED_DATASET_F_PATH, string(folder_name));
    end

    % Gathering the list of all matching files
    files = dir(string(strcat( ...
        MASTER_DATASET_F_PATH, ...
        folder_name, '\*', ...
        IMAGE_FILE_TYPE)));

    %% Looping through all the files
    for i = 1 : length(files)

        % Generating the actual file name
        old_file_name = strcat( ...
            MASTER_DATASET_F_PATH, ...
            folder_name, '\', ...
            files(i).name);

        % Creating the new file name for the processed data
        new_file_name = strcat( ...
            PROCESSED_DATASET_F_PATH, ...
            folder_name, '\', ...
            files(i).name);

        % Reading the image
        loaded_image = imread(string(old_file_name));

        % Re-shaping the image read
        resized_image = imresize(loaded_image, IMAGE_SHAPE);

        % Saving the image to the new destination
        imwrite(resized_image, string(new_file_name));
    end
end

