%% LOAD_DATA_DIRECT - Load and prepare FT-IR data with QC filtering
%
% Loads training and test data tables, applies quality control filtering,
% and packages data for classification pipeline. Works directly with original
% data tables without creating intermediate files.
%
% SYNTAX:
%   data = load_data_direct(cfg)
%
% INPUTS:
%   cfg - Configuration structure from config.m containing:
%         * paths.data: Path to data files
%         * paths.qc: Path to QC results
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
%          * wavenumbers: Wavenumber values for spectral features
%
% NOTES:
%   - Applies QC filtering if qc_flags.mat exists, otherwise uses all spectra
%   - Patient_ID used for patient-wise stratification in cross-validation
%   - Diss_ID identifies individual samples (some patients have multiple samples)
%   - Spectra remain in original CombinedSpectra cell format
%
% EXAMPLE:
%   cfg = config();
%   data = load_data_direct(cfg);
%
% See also: config, evaluate_test_set_direct

function data = load_data_direct(cfg)
    %% Load Raw Data Tables
    fprintf('Loading data tables...\n');
    load(fullfile(cfg.paths.data, 'data_table_train.mat'), 'dataTableTrain');
    load(fullfile(cfg.paths.data, 'data_table_test.mat'), 'dataTableTest');
    load(fullfile(cfg.paths.data, 'wavenumbers.mat'), 'wavenumbers_roi');
    
    %% Load QC Results (if available)
    qc_file = fullfile(cfg.paths.qc, 'qc_flags.mat');
    if exist(qc_file, 'file')
        fprintf('Loading QC results...\n');
        load(qc_file, 'qc_results');
    else
        fprintf('No QC results found - using all spectra\n');
        % Create dummy QC that accepts everything
        qc_results = create_dummy_qc(dataTableTrain, dataTableTest);
    end
    
    %% Process Training Data
    fprintf('Processing training data...\n');
    train = process_dataset(dataTableTrain, qc_results.train, 'train');
    
    %% Process Test Data
    fprintf('Processing test data...\n');
    test = process_dataset(dataTableTest, qc_results.test, 'test');
    
    %% Package Output
    data = struct();
    data.train = train;
    data.test = test;
    data.wavenumbers = wavenumbers_roi;
    
    %% Summary Statistics
    fprintf('\n=== Data Summary ===\n');
    fprintf('Training set:\n');
    fprintf('  Samples (Diss_IDs): %d\n', train.n_samples);
    fprintf('  Unique patients: %d\n', length(unique(train.patient_id)));
    fprintf('  WHO-1: %d samples, WHO-3: %d samples\n', ...
            sum(train.labels == 1), sum(train.labels == 3));
    fprintf('  Total spectra: %d\n', train.total_spectra);
    
    fprintf('Test set:\n');
    fprintf('  Samples (Diss_IDs): %d\n', test.n_samples);
    fprintf('  Unique patients: %d\n', length(unique(test.patient_id)));
    fprintf('  WHO-1: %d samples, WHO-3: %d samples\n', ...
            sum(test.labels == 1), sum(test.labels == 3));
    fprintf('  Total spectra: %d\n', test.total_spectra);
    fprintf('====================\n\n');
end


%% Helper: Process a dataset
function dataset = process_dataset(dataTable, qc, dataset_name)
    % Apply QC filtering
    valid_samples = ~qc.sample_metrics.Outlier_Flag;
    
    % Initialize output structure
    dataset = struct();
    dataset.n_samples = sum(valid_samples);
    dataset.diss_id = cell(dataset.n_samples, 1);      % Sample IDs
    dataset.patient_id = cell(dataset.n_samples, 1);   % Patient IDs (for stratification)
    dataset.spectra = cell(dataset.n_samples, 1);      % {N_samples x 1} cell, each [N_spectra x N_wavenumbers]
    dataset.labels = zeros(dataset.n_samples, 1);      % WHO grade (1 or 3)
    dataset.metadata = struct();                        % Additional info
    
    % Extract data from valid samples
    sample_idx = 1;
    total_spectra = 0;
    
    for i = 1:height(dataTable)
        if valid_samples(i)
            % Get QC-filtered spectra for this sample
            all_spectra = dataTable.CombinedSpectra{i};
            valid_mask = qc.valid_spectra_masks{i};
            filtered_spectra = all_spectra(valid_mask, :);
            
            if isempty(filtered_spectra)
                warning('Sample %d has no valid spectra after QC - skipping', i);
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
    end
    
    % Trim to actual size (in case some samples were skipped)
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


%% Helper: Create dummy QC (accepts all spectra)
function qc_results = create_dummy_qc(dataTableTrain, dataTableTest)
    qc_results = struct();
    
    % Training set
    qc_results.train.sample_metrics.Outlier_Flag = false(height(dataTableTrain), 1);
    qc_results.train.valid_spectra_masks = cell(height(dataTableTrain), 1);
    for i = 1:height(dataTableTrain)
        n_spectra = size(dataTableTrain.CombinedSpectra{i}, 1);
        qc_results.train.valid_spectra_masks{i} = true(n_spectra, 1);
    end
    
    % Test set
    qc_results.test.sample_metrics.Outlier_Flag = false(height(dataTableTest), 1);
    qc_results.test.valid_spectra_masks = cell(height(dataTableTest), 1);
    for i = 1:height(dataTableTest)
        n_spectra = size(dataTableTest.CombinedSpectra{i}, 1);
        qc_results.test.valid_spectra_masks{i} = true(n_spectra, 1);
    end
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
    
    % Check Patient_ID vs Diss_ID
    n_unique_patients = length(unique(dataset.patient_id));
    n_samples = dataset.n_samples;
    fprintf('    %d samples (Diss_IDs) from %d patients\n', n_samples, n_unique_patients);
    
    if n_samples > n_unique_patients
        fprintf('    Note: Some patients have multiple samples (recurrent tumors)\n');
    end
    
    fprintf('    ✓ Validation passed\n');
end
