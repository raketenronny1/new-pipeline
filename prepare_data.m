%% PREPARE_DATA.M - Data Preparation for WHO Meningioma Classification
%
% PURPOSE:
%   Cleans train/test data tables to contain only raw FTIR spectra
%   by removing preprocessed columns and calculating mean raw spectrum.
%
% ACTIONS PERFORMED:
%   1. Rename: CombinedRawSpectra → RawSpectra
%   2. Calculate: MeanRawSpectrum (mean across all spectra per sample)
%   3. Delete: CombinedRawSpectra (after renaming)
%   4. Delete: CombinedSpectra_PP1 (preprocessed)
%   5. Delete: CombinedSpectra_PP2 (preprocessed)
%   6. Delete: MeanSpectrum_PP1 (preprocessed)
%   7. Delete: MeanSpectrum_PP2 (preprocessed)
%   8. Validate: Train/test split matches split_info
%
% SOURCE FILES:
%   - data/data_table_train.mat (52 samples, 42 patients)
%   - data/data_table_test.mat (24 samples, 15 patients)
%   - data/dataset_complete.mat (115 samples, 57 patients - dataset_men variable)
%   - data/metadata_all_patients.mat (contains metadata_patients)
%   - data/split_info.mat (contains split information)
%
% CRITICAL FIX:
%   Previous pipeline used preprocessed spectra (PP1/PP2) as input,
%   causing double-preprocessing. This script ensures only raw spectra
%   are used as input to the preprocessing pipeline.
%
% OUTPUT FILES:
%   - data/data_table_train.mat (cleaned with RawSpectra + MeanRawSpectrum)
%   - data/data_table_test.mat (cleaned with RawSpectra + MeanRawSpectrum)
%   - data/dataset_complete.mat (cleaned dataset_men with RawSpectra + MeanRawSpectrum)
%   - data/backup_original/ (backup of original files)
%
% USAGE:
%   Run this script ONCE before running run_full_pipeline.m
%
% Author: AI Assistant
% Date: October 30, 2025

%% STEP 1.1: LOAD ALL DATA
fprintf('=== LOADING SOURCE FILES ===\n');

load('data/data_table_train.mat');          % -> data_table_train
load('data/data_table_test.mat');           % -> data_table_test
load('data/dataset_complete.mat');          % -> dataset_men
load('data/metadata_all_patients.mat');     % -> metadata_patients
load('data/split_info.mat');                % -> split_info

fprintf('✓ Loaded data_table_train (%d samples)\n', height(data_table_train));
fprintf('✓ Loaded data_table_test (%d samples)\n', height(data_table_test));
fprintf('✓ Loaded dataset_complete (%d samples)\n', height(dataset_men));
fprintf('✓ Loaded metadata_all_patients\n');
fprintf('✓ Loaded split_info\n\n');

%% STEP 1.2: CLEAN TRAIN TABLE
fprintf('=== CLEANING TRAIN TABLE ===\n');

% Rename CombinedRawSpectra to RawSpectra
data_table_train.RawSpectra = data_table_train.CombinedRawSpectra;
fprintf('✓ Renamed CombinedRawSpectra → RawSpectra\n');

% Calculate MeanRawSpectrum
% For each row, compute mean across all spectra (assuming cell array of matrices)
n_samples = height(data_table_train);
data_table_train.MeanRawSpectrum = cell(n_samples, 1);

for i = 1:n_samples
    raw_spectra = data_table_train.RawSpectra{i};
    % If multiple spectra exist (rows = spectra, cols = features), take mean
    data_table_train.MeanRawSpectrum{i} = mean(raw_spectra, 1);
end
fprintf('✓ Calculated MeanRawSpectrum for %d samples\n', n_samples);

% Remove preprocessed columns
data_table_train.CombinedRawSpectra = [];
data_table_train.CombinedSpectra_PP1 = [];
data_table_train.CombinedSpectra_PP2 = [];
data_table_train.MeanSpectrum_PP1 = [];
data_table_train.MeanSpectrum_PP2 = [];
fprintf('✓ Removed 5 preprocessed columns\n\n');

%% STEP 1.3: CLEAN TEST TABLE
fprintf('=== CLEANING TEST TABLE ===\n');

% Apply same logic to test table
data_table_test.RawSpectra = data_table_test.CombinedRawSpectra;
fprintf('✓ Renamed CombinedRawSpectra → RawSpectra\n');

% Calculate MeanRawSpectrum
n_samples = height(data_table_test);
data_table_test.MeanRawSpectrum = cell(n_samples, 1);

for i = 1:n_samples
    raw_spectra = data_table_test.RawSpectra{i};
    data_table_test.MeanRawSpectrum{i} = mean(raw_spectra, 1);
end
fprintf('✓ Calculated MeanRawSpectrum for %d samples\n', n_samples);

% Remove preprocessed columns
data_table_test.CombinedRawSpectra = [];
data_table_test.CombinedSpectra_PP1 = [];
data_table_test.CombinedSpectra_PP2 = [];
data_table_test.MeanSpectrum_PP1 = [];
data_table_test.MeanSpectrum_PP2 = [];
fprintf('✓ Removed 5 preprocessed columns\n\n');

%% STEP 1.4: CLEAN DATASET_MEN (COMPLETE DATASET)
fprintf('=== CLEANING DATASET_MEN (COMPLETE DATASET) ===\n');

% Apply same logic to dataset_men
dataset_men.RawSpectra = dataset_men.CombinedRawSpectra;
fprintf('✓ Renamed CombinedRawSpectra → RawSpectra\n');

% Calculate MeanRawSpectrum
n_samples = height(dataset_men);
dataset_men.MeanRawSpectrum = cell(n_samples, 1);

for i = 1:n_samples
    raw_spectra = dataset_men.RawSpectra{i};
    dataset_men.MeanRawSpectrum{i} = mean(raw_spectra, 1);
end
fprintf('✓ Calculated MeanRawSpectrum for %d samples\n', n_samples);

% Remove preprocessed columns
dataset_men.CombinedRawSpectra = [];
dataset_men.CombinedSpectra_PP1 = [];
dataset_men.CombinedSpectra_PP2 = [];
dataset_men.MeanSpectrum_PP1 = [];
dataset_men.MeanSpectrum_PP2 = [];
fprintf('✓ Removed 5 preprocessed columns\n\n');

%% STEP 1.5: VALIDATE SPLIT INTEGRITY
fprintf('=== VALIDATING SPLIT INTEGRITY ===\n');

% Get actual patient IDs from data tables
train_patients_actual = unique(data_table_train.Patient_ID);
test_patients_actual = unique(data_table_test.Patient_ID);
complete_patients_actual = unique(dataset_men.Patient_ID);

% Verify no overlap between train and test patients
overlap = intersect(train_patients_actual, test_patients_actual);
assert(isempty(overlap), 'ERROR: Patient overlap detected between train and test sets!');
fprintf('✓ No patient overlap between train and test\n');

% Verify train and test patients are subsets of complete dataset
all_split_patients = union(train_patients_actual, test_patients_actual);
assert(all(ismember(all_split_patients, complete_patients_actual)), ...
    'ERROR: Train/Test contains patients not in complete dataset!');
fprintf('✓ Train and test patients are subset of complete dataset\n');
fprintf('  Complete dataset: %d patients\n', length(complete_patients_actual));
fprintf('  Train + Test: %d patients (filtered subset)\n', length(all_split_patients));

% Verify counts match split_info
assert(length(train_patients_actual) == split_info.train_patients, ...
    'ERROR: Train patient count mismatch! Expected %d, found %d', ...
    split_info.train_patients, length(train_patients_actual));
assert(length(test_patients_actual) == split_info.test_patients, ...
    'ERROR: Test patient count mismatch! Expected %d, found %d', ...
    split_info.test_patients, length(test_patients_actual));

fprintf('✓ Patient counts match split_info: %d train, %d test\n', ...
    length(train_patients_actual), length(test_patients_actual));

% Verify sample counts match split_info
assert(height(data_table_train) == split_info.train_count, ...
    'ERROR: Train sample count mismatch! Expected %d, found %d', ...
    split_info.train_count, height(data_table_train));
assert(height(data_table_test) == split_info.test_count, ...
    'ERROR: Test sample count mismatch! Expected %d, found %d', ...
    split_info.test_count, height(data_table_test));

fprintf('✓ Sample counts match split_info: %d train, %d test\n\n', ...
    height(data_table_train), height(data_table_test));

%% STEP 1.6: DISPLAY SUMMARY
fprintf('=== DATA PREPARATION SUMMARY ===\n');
fprintf('Complete dataset: %d samples, %d patients\n', ...
    height(dataset_men), length(unique(dataset_men.Patient_ID)));
fprintf('Train set: %d samples, %d patients\n', ...
    height(data_table_train), length(unique(data_table_train.Patient_ID)));
fprintf('Test set: %d samples, %d patients\n', ...
    height(data_table_test), length(unique(data_table_test.Patient_ID)));

% Check spectral dimensions
sample_spectrum = data_table_train.MeanRawSpectrum{1};
fprintf('Spectral features: %d\n', length(sample_spectrum));

% Display class distribution
fprintf('\nTrain WHO Grade distribution:\n');
disp(tabulate(data_table_train.WHO_Grade));
fprintf('\nTest WHO Grade distribution:\n');
disp(tabulate(data_table_test.WHO_Grade));

fprintf('\n=== COLUMNS IN CLEANED TABLES ===\n');
fprintf('Train columns: %s\n', strjoin(data_table_train.Properties.VariableNames, ', '));
fprintf('\nExpected columns should include:\n');
fprintf('  ✓ RawSpectra (renamed from CombinedRawSpectra)\n');
fprintf('  ✓ MeanRawSpectrum (newly calculated)\n');
fprintf('  ✗ CombinedRawSpectra (removed)\n');
fprintf('  ✗ CombinedSpectra_PP1 (removed)\n');
fprintf('  ✗ CombinedSpectra_PP2 (removed)\n');
fprintf('  ✗ MeanSpectrum_PP1 (removed)\n');
fprintf('  ✗ MeanSpectrum_PP2 (removed)\n\n');

fprintf('Complete dataset columns: %s\n', strjoin(dataset_men.Properties.VariableNames, ', '));

%% STEP 1.7: SAVE CLEANED FILES
fprintf('=== SAVING CLEANED FILES ===\n');

% Backup original files first
if ~exist('data/backup_original', 'dir')
    mkdir('data/backup_original');
    fprintf('✓ Created backup directory\n');
end

try
    copyfile('data/data_table_train.mat', 'data/backup_original/data_table_train_original.mat');
    fprintf('✓ Backed up original train file\n');
catch ME
    fprintf('⚠ Could not backup train file (may already exist): %s\n', ME.message);
end

try
    copyfile('data/data_table_test.mat', 'data/backup_original/data_table_test_original.mat');
    fprintf('✓ Backed up original test file\n');
catch ME
    fprintf('⚠ Could not backup test file (may already exist): %s\n', ME.message);
end

try
    copyfile('data/dataset_complete.mat', 'data/backup_original/dataset_complete_original.mat');
    fprintf('✓ Backed up original complete dataset file\n');
catch ME
    fprintf('⚠ Could not backup complete dataset file (may already exist): %s\n', ME.message);
end

% Save cleaned versions
try
    save('data/data_table_train.mat', 'data_table_train', '-v7.3');
    fprintf('✓ Saved cleaned train file\n');
catch ME
    error('Failed to save train file: %s', ME.message);
end

try
    save('data/data_table_test.mat', 'data_table_test', '-v7.3');
    fprintf('✓ Saved cleaned test file\n');
catch ME
    error('Failed to save test file: %s', ME.message);
end

try
    save('data/dataset_complete.mat', 'dataset_men', '-v7.3');
    fprintf('✓ Saved cleaned complete dataset file\n');
catch ME
    error('Failed to save complete dataset file: %s', ME.message);
end

fprintf('\n✓ Cleaned data files saved\n');
fprintf('✓ Original files backed up to data/backup_original/\n');
fprintf('\n=== PREPARATION COMPLETE ===\n');
fprintf('You can now run run_full_pipeline.m\n');
