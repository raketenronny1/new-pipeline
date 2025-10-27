function [data_table_train, data_table_test, split_info] = split_train_test(dataset_men, varargin)
% SPLIT_TRAIN_TEST Split dataset into training and test sets
%
% Implements the train/test splitting strategy with the following rules:
%   1. Only WHO-1 and WHO-3 samples included (WHO-2 excluded)
%   2. All PATIENTS with methylation data → TEST set (all their samples)
%   3. Remaining PATIENTS split to achieve balanced WHO-1/WHO-3 in training
%   4. PATIENT-LEVEL SPLIT: All samples from one patient stay together
%   5. GOLDEN RULE: No ProbeUID overlap (same sample never in both sets)
%
% Syntax:
%   [data_table_train, data_table_test] = split_train_test(dataset_men)
%   [data_table_train, data_table_test, split_info] = split_train_test(dataset_men, 'Name', Value, ...)
%
% Inputs:
%   dataset_men - Table with FTIR data (from prepare_ftir_dataset)
%
% Optional Parameters:
%   'Verbose'      - Display progress messages (default: true)
%   'SaveOutput'   - Save train/test tables to data/ (default: true)
%   'OutputDir'    - Directory for output files (default: 'data')
%   'RandomSeed'   - Seed for reproducibility (default: 42)
%
% Outputs:
%   data_table_train - Training set table
%   data_table_test  - Test set table
%   split_info       - Struct with split statistics and validation results
%
% Example:
%   load('data/dataset_complete.mat', 'dataset_men');
%   [train_data, test_data, info] = split_train_test(dataset_men);
%
% See also: prepare_ftir_dataset

% Author: AI Assistant
% Date: 2025-10-24

%% Parse input arguments
p = inputParser;
addRequired(p, 'dataset_men', @(x) istable(x) && height(x) > 0);
addParameter(p, 'Verbose', true, @islogical);
addParameter(p, 'SaveOutput', true, @islogical);
addParameter(p, 'OutputDir', 'data', @(x) ischar(x) || isstring(x));
addParameter(p, 'RandomSeed', 42, @isnumeric);
parse(p, dataset_men, varargin{:});

verbose = p.Results.Verbose;
save_output = p.Results.SaveOutput;
output_dir = char(p.Results.OutputDir);
random_seed = p.Results.RandomSeed;

%% Initialize
rng(random_seed);  % For reproducible splits

if verbose
    fprintf('\n========================================================================\n');
    fprintf('  Train/Test Split Pipeline\n');
    fprintf('  Splitting Strategy: Balanced WHO-1/WHO-3, Methylation → Test\n');
    fprintf('========================================================================\n\n');
end

% Validate required columns
required_cols = {'ProbeUID', 'Patient_ID', 'WHO_Grade', 'CombinedSpectra_PP1', 'CombinedSpectra_PP2'};
missing_cols = setdiff(required_cols, dataset_men.Properties.VariableNames);
if ~isempty(missing_cols)
    error('Missing required columns: %s', strjoin(missing_cols, ', '));
end

initial_count = height(dataset_men);
if verbose
    fprintf('Step 1: Input dataset validation...\n');
    fprintf('  Total probes: %d\n', initial_count);
    fprintf('  Columns: %d\n', width(dataset_men));
end

%% Step 2: Filter to WHO-1 and WHO-3 only
if verbose
    fprintf('\nStep 2: Filtering to WHO-1 and WHO-3 samples...\n');
end

% Ensure WHO_Grade is categorical
if ~iscategorical(dataset_men.WHO_Grade)
    dataset_men.WHO_Grade = categorical(dataset_men.WHO_Grade);
end

% Filter for WHO-1 and WHO-3
is_who13 = ismember(dataset_men.WHO_Grade, {'WHO-1', 'WHO-3'});
dataset_filtered = dataset_men(is_who13, :);

if verbose
    fprintf('  WHO-1 samples: %d\n', sum(dataset_filtered.WHO_Grade == 'WHO-1'));
    fprintf('  WHO-3 samples: %d\n', sum(dataset_filtered.WHO_Grade == 'WHO-3'));
    fprintf('  WHO-2 excluded: %d\n', sum(dataset_men.WHO_Grade == 'WHO-2'));
    fprintf('  Filtered dataset: %d probes\n', height(dataset_filtered));
end

if height(dataset_filtered) == 0
    error('No WHO-1 or WHO-3 samples found in dataset');
end

%% Step 3: Identify methylation patients → TEST (except "mal" cluster → TRAIN)
if verbose
    fprintf('\nStep 3: Assigning methylation patients to test/train sets...\n');
end

% Find samples with methylation data
has_methylation = false(height(dataset_filtered), 1);
has_mal_cluster = false(height(dataset_filtered), 1);

if ismember('methylation_class', dataset_filtered.Properties.VariableNames)
    has_methylation = ~ismissing(dataset_filtered.methylation_class) & ...
                      dataset_filtered.methylation_class ~= '<undefined>';
end

if ismember('methylation_cluster', dataset_filtered.Properties.VariableNames)
    has_mal_cluster = ~ismissing(dataset_filtered.methylation_cluster) & ...
                      dataset_filtered.methylation_cluster == 'mal';
end

% Get patients with methylation data (excluding "mal" cluster)
has_non_mal_methylation = has_methylation & ~has_mal_cluster;
methylation_patient_ids = unique(dataset_filtered.Patient_ID(has_non_mal_methylation));

% Get patients with "mal" cluster (will go to TRAIN)
mal_cluster_patient_ids = unique(dataset_filtered.Patient_ID(has_mal_cluster));

n_methylation_samples = sum(has_non_mal_methylation);
n_methylation_patients = numel(methylation_patient_ids);
n_mal_samples = sum(has_mal_cluster);
n_mal_patients = numel(mal_cluster_patient_ids);

% Samples from non-mal methylation patients go to TEST
test_patient_mask = ismember(dataset_filtered.Patient_ID, methylation_patient_ids);
n_samples_from_meth_patients = sum(test_patient_mask);

if verbose
    fprintf('  Methylation cluster "mal" patients → TRAIN: %d patients (%d samples)\n', ...
            n_mal_patients, sum(ismember(dataset_filtered.Patient_ID, mal_cluster_patient_ids)));
    fprintf('  Other methylation patients → TEST: %d patients (%d methylation samples)\n', ...
            n_methylation_patients, n_methylation_samples);
    fprintf('  Total samples from test methylation patients: %d\n', n_samples_from_meth_patients);
end

%% Step 4: Create patient-level training pool
if verbose
    fprintf('\nStep 4: Creating patient-level training pool...\n');
end

% Training pool: patients without methylation data + "mal" cluster patients
% (Note: "mal" cluster patients are forced into training pool even though they have methylation data)
train_pool_mask = ~test_patient_mask | ismember(dataset_filtered.Patient_ID, mal_cluster_patient_ids);
train_pool = dataset_filtered(train_pool_mask, :);

% Group by patient and get sample counts per WHO grade for each patient
unique_patients = unique(train_pool.Patient_ID);
n_patients_available = numel(unique_patients);

% Build patient info: Patient_ID, WHO_Grade, n_samples
patient_info = table();
for i = 1:n_patients_available
    patient_id = unique_patients(i);
    patient_samples = train_pool(train_pool.Patient_ID == patient_id, :);
    
    % Determine dominant WHO grade for this patient (most common)
    who_grades = patient_samples.WHO_Grade;
    who1_count = sum(who_grades == 'WHO-1');
    who3_count = sum(who_grades == 'WHO-3');
    
    if who1_count > who3_count
        patient_who = categorical({'WHO-1'});
    elseif who3_count > who1_count
        patient_who = categorical({'WHO-3'});
    else
        % Tie: use first sample's grade
        patient_who = who_grades(1);
    end
    
    patient_info = [patient_info; table(patient_id, patient_who, height(patient_samples), ...
                                        'VariableNames', {'Patient_ID', 'WHO_Grade', 'n_samples'})];
end

% Separate patients by WHO grade
who1_patients = patient_info(patient_info.WHO_Grade == 'WHO-1', :);
who3_patients = patient_info(patient_info.WHO_Grade == 'WHO-3', :);

if verbose
    fprintf('  Available patients for training: %d\n', n_patients_available);
    fprintf('  WHO-1 patients: %d (total samples: %d)\n', ...
            height(who1_patients), sum(who1_patients.n_samples));
    fprintf('  WHO-3 patients: %d (total samples: %d)\n', ...
            height(who3_patients), sum(who3_patients.n_samples));
end

%% Step 5: Balance training set at SAMPLE level by selecting patients
if verbose
    fprintf('\nStep 5: Selecting patients for balanced training...\n');
end

% Strategy: Select patients to get as close as possible to equal WHO-1 and WHO-3 sample counts
if isempty(who1_patients) || isempty(who3_patients)
    warning('Class imbalance: one WHO grade has no patients for training');
    train_selected_patients = [];
else
    % Start with all patients from minority class
    total_who1_samples = sum(who1_patients.n_samples);
    total_who3_samples = sum(who3_patients.n_samples);
    
    if total_who1_samples <= total_who3_samples
        % WHO-1 is minority: take all WHO-1 patients
        train_who1_patients = who1_patients.Patient_ID;
        target_samples = total_who1_samples;
        
        % Select WHO-3 patients to match (approximately)
        who3_shuffled = who3_patients(randperm(height(who3_patients)), :);
        cumsum_samples = cumsum(who3_shuffled.n_samples);
        n_who3_to_take = find(cumsum_samples >= target_samples, 1);
        if isempty(n_who3_to_take)
            n_who3_to_take = height(who3_shuffled);
        end
        train_who3_patients = who3_shuffled.Patient_ID(1:n_who3_to_take);
    else
        % WHO-3 is minority: take all WHO-3 patients
        train_who3_patients = who3_patients.Patient_ID;
        target_samples = total_who3_samples;
        
        % Select WHO-1 patients to match (approximately)
        who1_shuffled = who1_patients(randperm(height(who1_patients)), :);
        cumsum_samples = cumsum(who1_shuffled.n_samples);
        n_who1_to_take = find(cumsum_samples >= target_samples, 1);
        if isempty(n_who1_to_take)
            n_who1_to_take = height(who1_shuffled);
        end
        train_who1_patients = who1_shuffled.Patient_ID(1:n_who1_to_take);
    end
    
    train_selected_patients = [train_who1_patients; train_who3_patients];
end

%% Step 6: Assemble train and test tables
if verbose
    fprintf('\nStep 6: Creating final train and test tables...\n');
end

% Training set: all samples from selected patients
is_train = ismember(dataset_filtered.Patient_ID, train_selected_patients);
data_table_train = dataset_filtered(is_train, :);

% Test set: all samples from methylation patients + remaining patients
is_test = ~is_train;
data_table_test = dataset_filtered(is_test, :);

% Fix nested cell arrays - unwrap spectral data columns
% When subsetting tables, MATLAB sometimes creates nested cells {1x1 cell} instead of the actual matrix
spectral_cols = {'CombinedRawSpectra', 'CombinedSpectra_PP1', 'CombinedSpectra_PP2', ...
                 'MeanSpectrum_PP1', 'MeanSpectrum_PP2'};

for col_idx = 1:length(spectral_cols)
    col_name = spectral_cols{col_idx};
    
    % Check if column exists in the tables
    if ismember(col_name, data_table_train.Properties.VariableNames)
        % Unwrap training data
        for row_idx = 1:height(data_table_train)
            cell_val = data_table_train.(col_name){row_idx};
            % If it's a nested cell {1x1 cell}, unwrap it
            if iscell(cell_val) && numel(cell_val) == 1
                data_table_train.(col_name){row_idx} = cell_val{1};
            end
        end
    end
    
    if ismember(col_name, data_table_test.Properties.VariableNames)
        % Unwrap test data
        for row_idx = 1:height(data_table_test)
            cell_val = data_table_test.(col_name){row_idx};
            % If it's a nested cell {1x1 cell}, unwrap it
            if iscell(cell_val) && numel(cell_val) == 1
                data_table_test.(col_name){row_idx} = cell_val{1};
            end
        end
    end
end

if verbose
    fprintf('  Final TRAIN: %d patients, %d samples\n', ...
            numel(unique(data_table_train.Patient_ID)), height(data_table_train));
    fprintf('  Final TEST: %d patients, %d samples\n', ...
            numel(unique(data_table_test.Patient_ID)), height(data_table_test));
end

%% Step 7: Validation checks
if verbose
    fprintf('\nStep 7: Validation checks...\n');
end

% Check 1: No overlap in ProbeUID (GOLDEN RULE)
train_ids = data_table_train.ProbeUID;
test_ids = data_table_test.ProbeUID;
overlap_ids = intersect(train_ids, test_ids);
if ~isempty(overlap_ids)
    error('CRITICAL: ProbeUID overlap detected (%d samples) - violates golden rule!', numel(overlap_ids));
end

% Check 2: No patient overlap (NEW REQUIREMENT - patient-level split)
train_patients = unique(data_table_train.Patient_ID);
test_patients = unique(data_table_test.Patient_ID);
overlap_patients = intersect(train_patients, test_patients);
if ~isempty(overlap_patients)
    error('CRITICAL: Patient_ID overlap detected (%d patients) - violates patient-level split!', numel(overlap_patients));
end

% Check 3: Class balance in training
if height(data_table_train) > 0
    train_who1_count = sum(data_table_train.WHO_Grade == 'WHO-1');
    train_who3_count = sum(data_table_train.WHO_Grade == 'WHO-3');
    is_balanced = (train_who1_count == train_who3_count);
else
    train_who1_count = 0;
    train_who3_count = 0;
    is_balanced = false;
end

% Check 4: Total count matches
total_split = height(data_table_train) + height(data_table_test);
count_matches = (total_split == height(dataset_filtered));

if verbose
    fprintf('  ✓ Train ProbeUIDs: %d unique\n', numel(unique(train_ids)));
    fprintf('  ✓ Test ProbeUIDs: %d unique\n', numel(unique(test_ids)));
    if isempty(overlap_ids)
        fprintf('  ✓ No ProbeUID overlap between train and test\n');
    else
        fprintf('  ✗ ProbeUID overlap: %d samples\n', numel(overlap_ids));
    end
    
    if isempty(overlap_patients)
        fprintf('  ✓ No Patient_ID overlap (patient-level split enforced)\n');
    else
        fprintf('  ✗ Patient_ID overlap: %d patients - VIOLATION!\n', numel(overlap_patients));
    end
    
    if is_balanced
        fprintf('  ✓ Training set balanced: %d WHO-1, %d WHO-3\n', ...
                train_who1_count, train_who3_count);
    else
        fprintf('  ✗ Training set imbalanced: %d WHO-1, %d WHO-3\n', ...
                train_who1_count, train_who3_count);
    end
    
    if count_matches
        fprintf('  ✓ Total count matches: %d (train) + %d (test) = %d (filtered)\n', ...
                height(data_table_train), height(data_table_test), height(dataset_filtered));
    else
        fprintf('  ✗ Count mismatch: %d total vs %d filtered\n', total_split, height(dataset_filtered));
    end
end

%% Step 8: Demographics summary
if verbose
    fprintf('\n========================================================================\n');
    fprintf('  Split Complete - Demographics Summary\n');
    fprintf('========================================================================\n\n');
    
    print_demographics(data_table_train, 'TRAINING');
    print_demographics(data_table_test, 'TEST');
end

%% Step 9: Create split info structure
split_info = struct();
split_info.initial_count = initial_count;
split_info.filtered_count = height(dataset_filtered);
split_info.methylation_samples = n_methylation_samples;
split_info.methylation_patients = n_methylation_patients;
split_info.train_count = height(data_table_train);
split_info.test_count = height(data_table_test);
split_info.train_who1 = train_who1_count;
split_info.train_who3 = train_who3_count;
split_info.train_patients = numel(unique(data_table_train.Patient_ID));
split_info.test_patients = numel(unique(data_table_test.Patient_ID));
split_info.is_balanced = is_balanced;
split_info.no_probe_overlap = isempty(overlap_ids);
split_info.no_patient_overlap = isempty(overlap_patients);
split_info.count_matches = count_matches;
split_info.random_seed = random_seed;

%% Step 10: Save output files
if save_output
    if verbose
        fprintf('\nStep 10: Saving output files...\n');
    end
    
    % Create output directory if needed
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Save train table
    train_file = fullfile(output_dir, 'data_table_train.mat');
    save(train_file, 'data_table_train', '-v7.3');
    if verbose
        fprintf('  Saved training set: %s\n', train_file);
    end
    
    % Save test table
    test_file = fullfile(output_dir, 'data_table_test.mat');
    save(test_file, 'data_table_test', '-v7.3');
    if verbose
        fprintf('  Saved test set: %s\n', test_file);
    end
    
    % Save split info
    info_file = fullfile(output_dir, 'split_info.mat');
    save(info_file, 'split_info', '-v7.3');
    if verbose
        fprintf('  Saved split info: %s\n', info_file);
    end
end

if verbose
    fprintf('\n========================================================================\n');
    fprintf('  Train/Test Split Complete!\n');
    fprintf('========================================================================\n\n');
end

end

%% Helper function: Print demographics
function print_demographics(data_table, set_name)
    fprintf('%s SET (n=%d)\n', set_name, height(data_table));
    if height(data_table) == 0
        fprintf('  (empty)\n\n');
        return;
    end
    
    % WHO Grade distribution
    if ismember('WHO_Grade', data_table.Properties.VariableNames)
        who_counts = countcats(data_table.WHO_Grade);
        who_cats = categories(data_table.WHO_Grade);
        fprintf('  WHO Grade: ');
        for i = 1:length(who_cats)
            if who_counts(i) > 0
                fprintf('%s=%d  ', who_cats{i}, who_counts(i));
            end
        end
        fprintf('\n');
    end
    
    % Age statistics
    if ismember('Age', data_table.Properties.VariableNames) && isnumeric(data_table.Age)
        valid_ages = data_table.Age(~isnan(data_table.Age) & ~isinf(data_table.Age));
        if ~isempty(valid_ages)
            fprintf('  Age: mean=%.1f±%.1f, range=[%.0f, %.0f]\n', ...
                    mean(valid_ages), std(valid_ages), min(valid_ages), max(valid_ages));
        end
    end
    
    % Sex distribution
    if ismember('Sex', data_table.Properties.VariableNames)
        sex_cats = categorical(data_table.Sex);
        sex_counts = countcats(sex_cats);
        sex_names = categories(sex_cats);
        fprintf('  Sex: ');
        for i = 1:length(sex_names)
            if sex_counts(i) > 0
                fprintf('%s=%d  ', sex_names{i}, sex_counts(i));
            end
        end
        fprintf('\n');
    end
    
    % Methylation count
    if ismember('methylation_class', data_table.Properties.VariableNames)
        has_meth = ~ismissing(data_table.methylation_class) & ...
                   data_table.methylation_class ~= '<undefined>';
        fprintf('  Methylation samples: %d\n', sum(has_meth));
    end
    
    fprintf('\n');
end
