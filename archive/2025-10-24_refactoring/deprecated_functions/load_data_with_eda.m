%% LOAD_DATA_WITH_EDA - Load data using EDA outlier detection results
%
% Loads training and test data tables, applies EDA-based outlier filtering,
% and packages data for classification pipeline. This replaces the old QC
% system with EDA's T²-Q outlier detection.
%
% SYNTAX:
%   data = load_data_with_eda(cfg)
%
% INPUTS:
%   cfg - Configuration structure from config.m containing:
%         * paths.data: Path to data files
%         * paths.eda: Path to EDA results
%
% OUTPUTS:
%   data - Structure containing:
%          * train: Training set structure with fields:
%            - spectra: Cell array of spectral matrices (one per sample)
%            - labels: WHO grade labels (1 or 3)
%            - diss_id: Sample IDs
%            - patient_id: Patient IDs (for stratification)
%            - n_samples: Number of samples
%            - total_spectra: Total number of spectra
%          * test: Test set structure (same fields as train)
%          * wavenumbers: Wavenumber values
%          * pca_model: PCA model from EDA (15 components)
%            - coeff: PCA coefficients (loadings)
%            - n_comp: Number of components (15)
%            - X_mean: Mean spectrum for centering
%            - explained: Variance explained by each PC
%
% WORKFLOW:
%   1. Load EDA results (outlier flags, PCA model)
%   2. Load train/test data tables
%   3. Filter out outlier spectra based on EDA T²-Q detection
%   4. Package data with PCA model for downstream use
%
% NOTES:
%   - Requires run_full_eda.m to be executed first
%   - PCA model from EDA is used for LDA classifier
%   - Other classifiers (PLS-DA, SVM, RF) use raw standardized spectra
%
% EXAMPLE:
%   cfg = config();
%   data = load_data_with_eda(cfg);
%
% See also: exploratory_data_analysis, run_full_eda, run_patientwise_cv_direct

function data = load_data_with_eda(cfg)
    fprintf('\n=== LOADING DATA WITH EDA-BASED OUTLIER FILTERING ===\n');
    
    %% Load EDA Results
    eda_file = fullfile(cfg.paths.eda, 'eda_results_PP1.mat');
    
    if ~exist(eda_file, 'file')
        error(['EDA results not found: %s\n' ...
               'Please run run_full_eda.m first to generate EDA results.'], eda_file);
    end
    
    fprintf('Loading EDA results...\n');
    load(eda_file, 'eda_results');
    
    % Extract outlier information
    outliers_both = eda_results.pca.outliers_both;  % Logical array for training spectra
    pca_info = eda_results.pca;
    
    % Use ProbeUIDs if available, otherwise fall back to indices
    if isfield(eda_results, 'probe_uids_pca')
        all_probe_ids_pca = eda_results.probe_uids_pca;  % Actual ProbeUIDs
    else
        all_probe_ids_pca = eda_results.probe_ids_pca;  % Fallback to indices
    end
    
    all_is_train = eda_results.is_train;  % Which spectra are from training set
    
    fprintf('  EDA outliers detected: %d / %d training spectra (%.1f%%)\n', ...
            sum(outliers_both), length(outliers_both), ...
            100*sum(outliers_both)/length(outliers_both));
    
    %% Load Data Tables
    fprintf('Loading data tables...\n');
    load(fullfile(cfg.paths.data, 'data_table_train.mat'), 'data_table_train');
    load(fullfile(cfg.paths.data, 'data_table_test.mat'), 'data_table_test');
    load(fullfile(cfg.paths.data, 'wavenumbers.mat'), 'wavenumbers_roi');
    
    %% Process Training Data with EDA Outlier Filtering
    fprintf('Processing training data with EDA filtering...\n');
    train = process_dataset_eda(data_table_train, outliers_both, all_probe_ids_pca, ...
                                 all_is_train, 'training');
    
    %% Process Test Data (no outlier filtering - evaluation set)
    fprintf('Processing test data (no filtering)...\n');
    test = process_dataset_no_filter(data_table_test, 'test');
    
    %% Package PCA Model from EDA
    % Use first 15 PCs as specified
    n_pcs_to_use = min(15, size(pca_info.coeff, 2));
    
    pca_model = struct();
    pca_model.coeff = pca_info.coeff(:, 1:n_pcs_to_use);
    pca_model.n_comp = n_pcs_to_use;
    pca_model.explained = pca_info.explained(1:n_pcs_to_use);
    pca_model.X_mean = eda_results.X_mean;  % Mean spectrum used for centering
    pca_model.total_variance = sum(pca_model.explained);
    
    fprintf('  PCA model: %d components (%.1f%% variance)\n', ...
            n_pcs_to_use, pca_model.total_variance);
    
    %% Package Output
    data = struct();
    data.train = train;
    data.test = test;
    data.wavenumbers = wavenumbers_roi;
    data.pca_model = pca_model;
    
    %% Summary Statistics
    fprintf('\n=== Data Summary ===\n');
    fprintf('Training set:\n');
    fprintf('  Samples (Diss_IDs): %d\n', train.n_samples);
    fprintf('  Unique patients: %d\n', length(unique(train.patient_id)));
    fprintf('  WHO-1: %d samples, WHO-3: %d samples\n', ...
            sum(train.labels == 1), sum(train.labels == 3));
    fprintf('  Total spectra (after outlier removal): %d\n', train.total_spectra);
    fprintf('  Spectra removed by EDA: %d\n', train.n_spectra_removed);
    
    fprintf('Test set:\n');
    fprintf('  Samples (Diss_IDs): %d\n', test.n_samples);
    fprintf('  Unique patients: %d\n', length(unique(test.patient_id)));
    fprintf('  WHO-1: %d samples, WHO-3: %d samples\n', ...
            sum(test.labels == 1), sum(test.labels == 3));
    fprintf('  Total spectra: %d\n', test.total_spectra);
    
    fprintf('PCA Model:\n');
    fprintf('  Components: %d PCs\n', pca_model.n_comp);
    fprintf('  Variance explained: %.2f%%\n', pca_model.total_variance);
    fprintf('====================\n\n');
end


%% Helper: Process training dataset with EDA outlier filtering
function dataset = process_dataset_eda(dataTable, outliers_both, all_probe_ids_pca, ...
                                       all_is_train, dataset_name)
    % Initialize output structure
    n_samples = height(dataTable);
    dataset = struct();
    dataset.diss_id = cell(n_samples, 1);
    dataset.patient_id = cell(n_samples, 1);
    dataset.spectra = cell(n_samples, 1);
    dataset.labels = zeros(n_samples, 1);
    dataset.metadata = struct();
    dataset.n_spectra_removed = 0;
    
    total_spectra = 0;
    sample_idx = 1;
    
    for i = 1:n_samples
        % Get all spectra for this sample
        all_spectra = dataTable.CombinedSpectra_PP1{i};
        n_spectra_sample = size(all_spectra, 1);
        
        % Find which spectra from this probe are in the PCA analysis
        probe_id = dataTable.ProbeUID(i);
        
        % Find indices of spectra from this probe in the PCA data
        probe_mask = (all_probe_ids_pca == probe_id) & all_is_train;
        
        if sum(probe_mask) == 0
            % This probe wasn't in training set for PCA (shouldn't happen)
            warning('Probe %d not found in EDA training data - keeping all spectra', probe_id);
            valid_mask = true(n_spectra_sample, 1);
        elseif sum(probe_mask) ~= n_spectra_sample
            warning(['Probe %d: spectrum count mismatch (%d in table, %d in EDA)\n' ...
                     'Using all spectra from table.'], ...
                    probe_id, n_spectra_sample, sum(probe_mask));
            valid_mask = true(n_spectra_sample, 1);
        else
            % Get outlier flags for this probe's spectra
            probe_outliers = outliers_both(probe_mask);
            valid_mask = ~probe_outliers;
            dataset.n_spectra_removed = dataset.n_spectra_removed + sum(probe_outliers);
        end
        
        % Filter spectra
        filtered_spectra = all_spectra(valid_mask, :);
        
        if isempty(filtered_spectra)
            warning('Sample %d has no valid spectra after EDA filtering - skipping', i);
            continue;
        end
        
        % Store sample information
        dataset.diss_id{sample_idx} = dataTable.Diss_ID{i};
        dataset.patient_id{sample_idx} = char(dataTable.Patient_ID(i));
        dataset.spectra{sample_idx} = filtered_spectra;
        
        % Extract WHO grade label
        label_val = dataTable.WHO_Grade(i);
        if iscategorical(label_val)
            label_str = char(label_val);
        else
            label_str = label_val;
        end
        
        if contains(label_str, '1')
            dataset.labels(sample_idx) = 1;
        elseif contains(label_str, '3')
            dataset.labels(sample_idx) = 3;
        else
            error('Unexpected WHO grade: %s', label_str);
        end
        
        % Store metadata
        dataset.metadata.age{sample_idx} = dataTable.Age(i);
        dataset.metadata.sex{sample_idx} = char(dataTable.Sex(i));
        
        total_spectra = total_spectra + size(filtered_spectra, 1);
        sample_idx = sample_idx + 1;
    end
    
    % Trim to actual size
    actual_size = sample_idx - 1;
    dataset.n_samples = actual_size;
    dataset.diss_id = dataset.diss_id(1:actual_size);
    dataset.patient_id = dataset.patient_id(1:actual_size);
    dataset.spectra = dataset.spectra(1:actual_size);
    dataset.labels = dataset.labels(1:actual_size);
    dataset.total_spectra = total_spectra;
    
    % Validation
    validate_dataset(dataset, dataset_name);
end


%% Helper: Process test dataset without filtering
function dataset = process_dataset_no_filter(dataTable, dataset_name)
    % Initialize output structure
    n_samples = height(dataTable);
    dataset = struct();
    dataset.diss_id = cell(n_samples, 1);
    dataset.patient_id = cell(n_samples, 1);
    dataset.spectra = cell(n_samples, 1);
    dataset.labels = zeros(n_samples, 1);
    dataset.metadata = struct();
    
    total_spectra = 0;
    sample_idx = 1;
    
    for i = 1:n_samples
        % Get all spectra (no filtering for test set)
        all_spectra = dataTable.CombinedSpectra_PP1{i};
        
        if isempty(all_spectra)
            warning('Sample %d has no spectra - skipping', i);
            continue;
        end
        
        % Store sample information
        dataset.diss_id{sample_idx} = dataTable.Diss_ID{i};
        dataset.patient_id{sample_idx} = char(dataTable.Patient_ID(i));
        dataset.spectra{sample_idx} = all_spectra;
        
        % Extract WHO grade label
        label_val = dataTable.WHO_Grade(i);
        if iscategorical(label_val)
            label_str = char(label_val);
        else
            label_str = label_val;
        end
        
        if contains(label_str, '1')
            dataset.labels(sample_idx) = 1;
        elseif contains(label_str, '3')
            dataset.labels(sample_idx) = 3;
        else
            error('Unexpected WHO grade: %s', label_str);
        end
        
        % Store metadata
        dataset.metadata.age{sample_idx} = dataTable.Age(i);
        dataset.metadata.sex{sample_idx} = char(dataTable.Sex(i));
        
        total_spectra = total_spectra + size(all_spectra, 1);
        sample_idx = sample_idx + 1;
    end
    
    % Trim to actual size
    actual_size = sample_idx - 1;
    dataset.n_samples = actual_size;
    dataset.diss_id = dataset.diss_id(1:actual_size);
    dataset.patient_id = dataset.patient_id(1:actual_size);
    dataset.spectra = dataset.spectra(1:actual_size);
    dataset.labels = dataset.labels(1:actual_size);
    dataset.total_spectra = total_spectra;
    
    % Validation
    validate_dataset(dataset, dataset_name);
end


%% Helper: Validate dataset
function validate_dataset(dataset, name)
    fprintf('  Validating %s dataset...\n', name);
    
    % Check for empty data
    assert(dataset.n_samples > 0, 'No samples in dataset');
    assert(~isempty(dataset.spectra), 'No spectra in dataset');
    
    % Check dimensions
    n_wavenumbers = size(dataset.spectra{1}, 2);
    for i = 1:dataset.n_samples
        assert(size(dataset.spectra{i}, 2) == n_wavenumbers, ...
               'Inconsistent wavenumber dimension in sample %d', i);
        assert(size(dataset.spectra{i}, 1) > 0, ...
               'Empty spectra in sample %d', i);
    end
    
    % Check for NaN/Inf
    for i = 1:dataset.n_samples
        assert(~any(isnan(dataset.spectra{i}(:))), ...
               'NaN values in sample %d', i);
        assert(~any(isinf(dataset.spectra{i}(:))), ...
               'Inf values in sample %d', i);
    end
    
    % Check labels
    unique_labels = unique(dataset.labels);
    assert(all(ismember(unique_labels, [1, 3])), ...
           'Invalid labels found (expected 1 or 3)');
    
    fprintf('    ✓ Validation passed\n');
end
