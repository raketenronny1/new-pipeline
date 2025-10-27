% Run EDA without visualizations (for batch mode)
% This version only computes PCA and outlier flags, skips plots

addpath('src/meningioma_ftir_pipeline');

fprintf('Loading training data only (to reduce memory usage)...\n');

% Load only training data for EDA
if ~exist('data/data_table_train.mat', 'file')
    error(['Training data file not found!\n' ...
           'Please run split_train_test first to generate data_table_train.mat']);
end

% Load training data using matfile to save memory
fprintf('  Loading train data...\n');
m_train = matfile('data/data_table_train.mat');
data_table_train = m_train.data_table_train;

fprintf('  Training set: %d probes (WHO-1 & WHO-3)\n', height(data_table_train));

% For EDA, we only need training data
dataset_men = data_table_train;
train_indices = true(height(dataset_men), 1);

fprintf('  Dataset ready: %d probes total\n', height(dataset_men));
fprintf('  All marked as training for PCA\n\n');

% Start timer
tic;

% Run EDA with training set specified and NO PLOTS
fprintf('Starting EDA (PCA + outlier detection, no plots)...\n\n');

% Call EDA with special flag to skip plots
eda_results = exploratory_data_analysis_no_plots(dataset_men, train_indices);

elapsed = toc;
fprintf('\n\nTotal time: %.1f seconds (%.1f minutes)\n', elapsed, elapsed/60);
fprintf('EDA completed successfully!\n');
fprintf('Results saved to: results/eda/eda_results_PP1.mat\n');
