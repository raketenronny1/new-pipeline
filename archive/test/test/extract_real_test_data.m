%% Script to extract representative spectra for test data
% This script extracts real spectra from the main dataset to create
% a smaller, representative test dataset

% Add all required directories to path
addpath(genpath(fileparts(fileparts(fileparts(mfilename('fullpath'))))));

% Load main datasets
fprintf('Loading original data files...\n');
try
    main_data_path = fullfile(fileparts(fileparts(fileparts(fileparts(mfilename('fullpath'))))), 'data');
    load(fullfile(main_data_path, 'data_table_train.mat'), 'dataTableTrain');
    load(fullfile(main_data_path, 'wavenumbers.mat'), 'wavenumbers_roi');
    fprintf('Data loaded successfully.\n');
catch ME
    fprintf('Error loading data: %s\n', ME.message);
    return;
end

% Display dataset info
fprintf('\n=== Dataset Information ===\n');
fprintf('Number of samples: %d\n', height(dataTableTrain));

% Check if WHO_Grade is categorical and convert if needed
if iscategorical(dataTableTrain.WHO_Grade)
    who_grades = double(dataTableTrain.WHO_Grade);
else
    who_grades = dataTableTrain.WHO_Grade;
end

size_who1 = sum(who_grades == 1);
size_who3 = sum(who_grades == 3);
fprintf('WHO-1 samples: %d\n', size_who1);
fprintf('WHO-3 samples: %d\n', size_who3);

% Check first sample's spectra
if ~isempty(dataTableTrain.CombinedSpectra{1})
    [n_spectra, n_wavenumbers] = size(dataTableTrain.CombinedSpectra{1});
    fprintf('First sample has %d spectra with %d wavenumbers each\n', n_spectra, n_wavenumbers);
else
    fprintf('First sample has no spectra\n');
end

% Create smaller test dataset
n_samples_who1 = 10; % Number of WHO-1 samples for test
n_samples_who3 = 10; % Number of WHO-3 samples for test
n_spectra_per_sample = 5; % Number of spectra to keep per sample

fprintf('\nExtracting representative spectra...\n');

% Get indices of WHO-1 and WHO-3 samples
if iscategorical(dataTableTrain.WHO_Grade)
    who1_indices = find(double(dataTableTrain.WHO_Grade) == 1);
    who3_indices = find(double(dataTableTrain.WHO_Grade) == 3);
else
    who1_indices = find(dataTableTrain.WHO_Grade == 1);
    who3_indices = find(dataTableTrain.WHO_Grade == 3);
end

% Select random samples (with seed for reproducibility)
rng(42);
if length(who1_indices) >= n_samples_who1 && length(who3_indices) >= n_samples_who3
    selected_who1 = who1_indices(randperm(length(who1_indices), n_samples_who1));
    selected_who3 = who3_indices(randperm(length(who3_indices), n_samples_who3));
    selected_indices = [selected_who1; selected_who3];
else
    fprintf('Not enough samples of each class. Using all available.\n');
    selected_indices = [who1_indices; who3_indices];
    n_samples_who1 = length(who1_indices);
    n_samples_who3 = length(who3_indices);
end

% Extract data for selected samples
test_table = dataTableTrain(selected_indices, :);

% Reduce number of spectra per sample
for i = 1:height(test_table)
    spectra = test_table.CombinedSpectra{i};
    if ~isempty(spectra)
        [current_n_spectra, ~] = size(spectra);
        if current_n_spectra > n_spectra_per_sample
            % Randomly select spectra
            selected_spectra_idx = randperm(current_n_spectra, n_spectra_per_sample);
            test_table.CombinedSpectra{i} = spectra(selected_spectra_idx, :);
        end
    else
        fprintf('Warning: Sample %d has no spectra\n', i);
    end
end

% Split into training and test sets
n_train = round(0.7 * height(test_table));
n_test = height(test_table) - n_train;

% Ensure balanced classes in both sets
n_train_who1 = round(n_samples_who1 * 0.7);
n_train_who3 = n_train - n_train_who1;

if iscategorical(test_table.WHO_Grade)
    who1_rows = double(test_table.WHO_Grade) == 1;
    who3_rows = double(test_table.WHO_Grade) == 3;
else
    who1_rows = test_table.WHO_Grade == 1;
    who3_rows = test_table.WHO_Grade == 3;
end

% Get indices for training and test
train_who1_idx = find(who1_rows, n_train_who1);
train_who3_idx = find(who3_rows, n_train_who3);
train_idx = [train_who1_idx; train_who3_idx];

% Test indices are the remaining rows
all_idx = 1:height(test_table);
test_idx = setdiff(all_idx, train_idx);

% Create final training and test tables
dataTableTrain_test = test_table(train_idx, :);
dataTableTest_test = test_table(test_idx, :);

% Save files to test data directory
test_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(test_dir, 'data');

if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end

fprintf('\nSaving test dataset files...\n');
% Rename the variables to what the pipeline expects
dataTableTrain = dataTableTrain_test;
dataTableTest = dataTableTest_test;
save(fullfile(data_dir, 'data_table_train.mat'), 'dataTableTrain');
save(fullfile(data_dir, 'data_table_test.mat'), 'dataTableTest');
save(fullfile(data_dir, 'wavenumbers.mat'), 'wavenumbers_roi');

fprintf('Done. Files saved to: %s\n', data_dir);
fprintf('\nSummary of created datasets:\n');
if iscategorical(dataTableTrain_test.WHO_Grade)
    train_who1_count = sum(double(dataTableTrain_test.WHO_Grade) == 1);
    train_who3_count = sum(double(dataTableTrain_test.WHO_Grade) == 3);
    test_who1_count = sum(double(dataTableTest_test.WHO_Grade) == 1);
    test_who3_count = sum(double(dataTableTest_test.WHO_Grade) == 3);
else
    train_who1_count = sum(dataTableTrain_test.WHO_Grade == 1);
    train_who3_count = sum(dataTableTrain_test.WHO_Grade == 3);
    test_who1_count = sum(dataTableTest_test.WHO_Grade == 1);
    test_who3_count = sum(dataTableTest_test.WHO_Grade == 3);
end

fprintf('Training set: %d samples (%d WHO-1, %d WHO-3)\n', ...
    height(dataTableTrain_test), train_who1_count, train_who3_count);
fprintf('Test set: %d samples (%d WHO-1, %d WHO-3)\n', ...
    height(dataTableTest_test), test_who1_count, test_who3_count);