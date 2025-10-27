% Run full EDA with detailed output
% NOTE: Run this in MATLAB GUI, not in batch mode, for large datasets
addpath('src/meningioma_ftir_pipeline');

fprintf('Loading training data only (to reduce memory usage)...\n');

% Load only training data for EDA
if ~exist('data/data_table_train.mat', 'file')
    error(['Training data file not found!\n' ...
           'Please run split_train_test first to generate data_table_train.mat']);
end

% Load training data
fprintf('  Loading train data...\n');
m_train = matfile('data/data_table_train.mat');
data_table_train = m_train.data_table_train;

fprintf('  Training set: %d probes (WHO-1 & WHO-3)\n', height(data_table_train));

% For EDA, we only need training data since:
% 1. PCA is computed on training data only
% 2. Outlier detection is for training data only
% 3. Test set is never modified
dataset_men = data_table_train;

% All samples are training samples
train_indices = true(height(dataset_men), 1);

fprintf('  Dataset ready: %d probes total\n', height(dataset_men));
fprintf('  All marked as training for PCA\n\n');

% Start timer
tic;

% Run EDA with training set specified
fprintf('Starting EDA (this may take several minutes)...\n\n');
eda_results = exploratory_data_analysis(dataset_men, ...
    'PreprocessingType', 'PP1', ...
    'Verbose', true, ...
    'TrainIndices', train_indices);

elapsed = toc;
fprintf('\n\nTotal time: %.1f seconds (%.1f minutes)\n', elapsed, elapsed/60);
fprintf('EDA completed successfully!\n');
fprintf('\nGenerated files in results/eda/:\n');

% List all generated files
files = dir('results/eda/*.png');
for i = 1:length(files)
    fprintf('  %2d. %s (%.1f KB)\n', i, files(i).name, files(i).bytes/1024);
end
