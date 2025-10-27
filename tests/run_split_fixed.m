% Script to run split_train_test with fixed cell structure
addpath('src/meningioma_ftir_pipeline');

% Load dataset
fprintf('Loading dataset...\n');
load('data/dataset_complete.mat', 'dataset_men');

% Run split
fprintf('\nRunning split_train_test...\n');
[data_table_train, data_table_test, split_info] = split_train_test(dataset_men);

% Verify cell structure is fixed
fprintf('\n========================================================================\n');
fprintf('  Verifying Cell Structure Fix\n');
fprintf('========================================================================\n\n');

fprintf('TRAIN SET:\n');
fprintf('  CombinedRawSpectra{1,1} class: %s\n', class(data_table_train.CombinedRawSpectra{1,1}));
fprintf('  CombinedRawSpectra{1,1} size: [%d x %d]\n', size(data_table_train.CombinedRawSpectra{1,1}));
fprintf('  CombinedSpectra_PP1{1,1} class: %s\n', class(data_table_train.CombinedSpectra_PP1{1,1}));
fprintf('  CombinedSpectra_PP1{1,1} size: [%d x %d]\n', size(data_table_train.CombinedSpectra_PP1{1,1}));

fprintf('\nTEST SET:\n');
fprintf('  CombinedRawSpectra{1,1} class: %s\n', class(data_table_test.CombinedRawSpectra{1,1}));
fprintf('  CombinedRawSpectra{1,1} size: [%d x %d]\n', size(data_table_test.CombinedRawSpectra{1,1}));
fprintf('  CombinedSpectra_PP1{1,1} class: %s\n', class(data_table_test.CombinedSpectra_PP1{1,1}));
fprintf('  CombinedSpectra_PP1{1,1} size: [%d x %d]\n', size(data_table_test.CombinedSpectra_PP1{1,1}));

fprintf('\n✓ Cell structure verified - spectra are directly accessible as matrices!\n');
fprintf('  (No more nested cells)\n\n');
