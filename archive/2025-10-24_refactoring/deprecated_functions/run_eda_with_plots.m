% Run EDA with plots (headless/batch-safe mode)
% Creates all visualizations without opening GUI windows

addpath('src/meningioma_ftir_pipeline');

fprintf('Loading training data...\n');

% Load training data
if ~exist('data/data_table_train.mat', 'file')
    error('Training data file not found! Run split_train_test first.');
end

m_train = matfile('data/data_table_train.mat');
data_table_train = m_train.data_table_train;

fprintf('  Training set: %d probes\n', height(data_table_train));

dataset_men = data_table_train;
train_indices = true(height(dataset_men), 1);

fprintf('  Dataset ready: %d probes\n\n', height(dataset_men));

% Start timer
tic;

% Run EDA with plots enabled
fprintf('Starting EDA with visualization generation...\n\n');
eda_results = exploratory_data_analysis(dataset_men, ...
    'PreprocessingType', 'PP1', ...
    'Verbose', true, ...
    'TrainIndices', train_indices, ...
    'Headless', true);  % Batch-safe plotting

elapsed = toc;
fprintf('\n\nTotal time: %.1f seconds (%.1f minutes)\n', elapsed, elapsed/60);
fprintf('EDA completed successfully!\n');
fprintf('\nGenerated files in results/eda/:\n');

% List all generated files
files = dir('results/eda/*.png');
for i = 1:length(files)
    fprintf('  %2d. %s (%.1f KB)\n', i, files(i).name, files(i).bytes/1024);
end

fprintf('\nMAT file:\n');
mat_file = dir('results/eda/eda_results_PP1.mat');
if ~isempty(mat_file)
    fprintf('  eda_results_PP1.mat (%.1f MB)\n', mat_file.bytes/1024/1024);
end
