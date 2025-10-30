%% TEST_DATA_LOADER - Unit tests for DataLoader class
%
% Tests:
%   1. Load complete dataset
%   2. Load train/test split
%   3. Data validation
%   4. Metadata computation
%   5. Patient-level statistics
%
% USAGE:
%   run test_data_loader.m

function test_data_loader()
    fprintf('=== Testing DataLoader Class ===\n\n');
    
    % Setup paths
    data_dir = 'data/';
    
    %% Test 1: Load Training Data
    fprintf('Test 1: Load training data... ');
    train_file = fullfile(data_dir, 'data_table_train.mat');
    
    if ~exist(train_file, 'file')
        fprintf('SKIPPED (file not found)\n');
    else
        [X_train, y_train, pids_train, meta_train] = ...
            DataLoader.load(train_file, 'Verbose', false);
        
        % Validate outputs
        assert(~isempty(X_train), 'X_train is empty');
        assert(~isempty(y_train), 'y_train is empty');
        assert(~isempty(pids_train), 'pids_train is empty');
        assert(isstruct(meta_train), 'metadata must be struct');
        
        % Check dimensions
        assert(size(X_train, 1) == length(y_train), 'Dimension mismatch');
        assert(size(X_train, 1) == length(pids_train), 'Patient ID mismatch');
        
        fprintf('✓ PASSED (%d samples, %d features)\n', ...
            size(X_train, 1), size(X_train, 2));
    end
    
    %% Test 2: Load Test Data
    fprintf('Test 2: Load test data... ');
    test_file = fullfile(data_dir, 'data_table_test.mat');
    
    if ~exist(test_file, 'file')
        fprintf('SKIPPED (file not found)\n');
    else
        [X_test, y_test, pids_test, meta_test] = ...
            DataLoader.load(test_file, 'Verbose', false);
        
        assert(~isempty(X_test), 'X_test is empty');
        assert(size(X_test, 1) == length(y_test), 'Dimension mismatch');
        
        fprintf('✓ PASSED (%d samples, %d features)\n', ...
            size(X_test, 1), size(X_test, 2));
    end
    
    %% Test 3: Load Split
    fprintf('Test 3: Load train/test split... ');
    
    if ~exist(train_file, 'file') || ~exist(test_file, 'file')
        fprintf('SKIPPED (files not found)\n');
    else
        [data_train, data_test, meta_split] = ...
            DataLoader.loadSplit(data_dir, 'Verbose', false);
        
        % Validate structure
        assert(isfield(data_train, 'X'), 'Missing X in train');
        assert(isfield(data_train, 'y'), 'Missing y in train');
        assert(isfield(data_train, 'patient_ids'), 'Missing patient_ids in train');
        
        % Validate no patient overlap
        overlap = intersect(unique(data_train.patient_ids), ...
                           unique(data_test.patient_ids));
        assert(isempty(overlap), 'Patient overlap detected!');
        
        fprintf('✓ PASSED (no patient overlap)\n');
    end
    
    %% Test 4: Metadata Computation
    fprintf('Test 4: Metadata computation... ');
    
    if exist(train_file, 'file')
        [~, ~, ~, meta] = DataLoader.load(train_file, 'Verbose', false);
        
        % Check required fields
        assert(isfield(meta, 'n_samples'), 'Missing n_samples');
        assert(isfield(meta, 'n_features'), 'Missing n_features');
        assert(isfield(meta, 'n_patients'), 'Missing n_patients');
        assert(isfield(meta, 'class_distribution'), 'Missing class_distribution');
        assert(isfield(meta, 'samples_per_patient'), 'Missing samples_per_patient');
        
        % Validate values
        assert(meta.n_samples > 0, 'Invalid n_samples');
        assert(meta.n_features > 0, 'Invalid n_features');
        assert(meta.n_patients > 0, 'Invalid n_patients');
        
        fprintf('✓ PASSED\n');
    else
        fprintf('SKIPPED (file not found)\n');
    end
    
    %% Test 5: Categorical Labels
    fprintf('Test 5: Categorical labels... ');
    
    if exist(train_file, 'file')
        [~, y, ~, ~] = DataLoader.load(train_file, 'Verbose', false);
        
        % Check categorical
        assert(iscategorical(y), 'Labels must be categorical');
        assert(length(unique(y)) >= 2, 'Must have at least 2 classes');
        
        fprintf('✓ PASSED (%d classes)\n', length(unique(y)));
    else
        fprintf('SKIPPED (file not found)\n');
    end
    
    %% Test 6: Patient IDs
    fprintf('Test 6: Patient IDs... ');
    
    if exist(train_file, 'file')
        [X, ~, pids, meta] = DataLoader.load(train_file, 'Verbose', false);
        
        % Check patient counts
        n_unique = length(unique(pids));
        assert(n_unique == meta.n_patients, 'Patient count mismatch');
        assert(n_unique < size(X, 1), 'Should have fewer patients than samples');
        
        fprintf('✓ PASSED (%d patients for %d samples)\n', n_unique, size(X, 1));
    else
        fprintf('SKIPPED (file not found)\n');
    end
    
    %% Test 7: Different Spectra Fields
    fprintf('Test 7: Different spectra fields... ');
    
    if exist(train_file, 'file')
        % Try PP1
        [X_pp1, ~, ~, ~] = DataLoader.load(train_file, ...
            'SpectraField', 'CombinedSpectra_PP1', 'Verbose', false);
        
        % Try PP2
        [X_pp2, ~, ~, ~] = DataLoader.load(train_file, ...
            'SpectraField', 'CombinedSpectra_PP2', 'Verbose', false);
        
        % Should have same number of samples but potentially different features
        assert(size(X_pp1, 1) == size(X_pp2, 1), 'Sample count mismatch');
        
        fprintf('✓ PASSED (PP1: %d, PP2: %d features)\n', ...
            size(X_pp1, 2), size(X_pp2, 2));
    else
        fprintf('SKIPPED (file not found)\n');
    end
    
    %% Test 8: Aggregation Methods
    fprintf('Test 8: Aggregation methods... ');
    
    if exist(train_file, 'file')
        [X_mean, ~, ~, ~] = DataLoader.load(train_file, ...
            'AggregationMethod', 'mean', 'Verbose', false);
        
        [X_median, ~, ~, ~] = DataLoader.load(train_file, ...
            'AggregationMethod', 'median', 'Verbose', false);
        
        % Should have same dimensions
        assert(isequal(size(X_mean), size(X_median)), ...
            'Aggregation methods produced different sizes');
        
        fprintf('✓ PASSED\n');
    else
        fprintf('SKIPPED (file not found)\n');
    end
    
    fprintf('\n=== ALL TESTS COMPLETED ===\n');
end
