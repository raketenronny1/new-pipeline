%% Direct Test of EDA Components
% Test each step of the EDA process separately

clear; close all; clc;

fprintf('\n=== Testing EDA Components ===\n\n');

%% Setup
addpath('src/meningioma_ftir_pipeline');
addpath('src/utils');
addpath('src/preprocessing');

%% Step 1: Load data
fprintf('Step 1: Loading training data...\n');
try
    m_train = matfile('data/data_table_train.mat');
    data_table_train = m_train.data_table_train;
    fprintf('  ✓ Loaded: %d x %d table\n', height(data_table_train), width(data_table_train));
catch ME
    fprintf('  ✗ FAILED: %s\n', ME.message);
    return;
end

%% Step 2: Check required columns
fprintf('\nStep 2: Checking required columns...\n');
required_cols = {'ProbeUID', 'Patient_ID', 'WHO_Grade', 'CombinedSpectra_PP1'};
for i = 1:length(required_cols)
    if ismember(required_cols{i}, data_table_train.Properties.VariableNames)
        fprintf('  ✓ %s\n', required_cols{i});
    else
        fprintf('  ✗ %s MISSING\n', required_cols{i});
        return;
    end
end

%% Step 3: Load wavenumbers
fprintf('\nStep 3: Loading wavenumbers...\n');
try
    load('data/wavenumbers.mat', 'wavenumbers');
    fprintf('  ✓ Loaded: %d wavenumbers\n', length(wavenumbers));
catch ME
    fprintf('  ✗ FAILED: %s\n', ME.message);
    return;
end

%% Step 4: Check spectra structure
fprintf('\nStep 4: Checking spectra structure...\n');
try
    first_sample_spectra = data_table_train.CombinedSpectra_PP1{1};
    [n_spectra, n_features] = size(first_sample_spectra);
    fprintf('  ✓ First sample has %d spectra with %d features\n', n_spectra, n_features);
    
    if n_features == length(wavenumbers)
        fprintf('  ✓ Features match wavenumbers\n');
    else
        fprintf('  ⚠ Feature mismatch: %d vs %d wavenumbers\n', n_features, length(wavenumbers));
    end
catch ME
    fprintf('  ✗ FAILED: %s\n', ME.message);
    return;
end

%% Step 5: Test EDA function exists and can be called
fprintf('\nStep 5: Checking EDA function...\n');
if exist('exploratory_data_analysis_no_plots', 'file')
    fprintf('  ✓ exploratory_data_analysis_no_plots found\n');
else
    fprintf('  ✗ Function not found\n');
    return;
end

%% Step 6: Try calling EDA on small subset
fprintf('\nStep 6: Testing EDA on first 5 samples...\n');
try
    dataset_small = data_table_train(1:5, :);
    train_indices = true(height(dataset_small), 1);
    
    fprintf('  Running EDA (this may take a moment)...\n');
    eda_results = exploratory_data_analysis_no_plots(dataset_small, train_indices);
    
    fprintf('  ✓ EDA completed successfully!\n');
    fprintf('  Outliers detected: %d\n', sum(eda_results.is_outlier));
    
catch ME
    fprintf('  ✗ FAILED: %s\n', ME.message);
    fprintf('  Error ID: %s\n', ME.identifier);
    if ~isempty(ME.stack)
        fprintf('  Location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    end
    return;
end

%% Summary
fprintf('\n=================================\n');
fprintf('✓ ALL COMPONENTS WORKING\n');
fprintf('=================================\n\n');
fprintf('EDA pipeline is functional.\n');
fprintf('You can now run: run_eda(''CreatePlots'', false)\n\n');
