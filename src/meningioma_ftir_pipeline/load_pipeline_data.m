%% LOAD_PIPELINE_DATA - Unified data loader for FTIR classification pipeline
%
% Loads training and test data with flexible outlier filtering options.
% This function consolidates the functionality of load_data_direct.m,
% load_data_with_eda.m, and load_and_prepare_data.m into a single interface.
%
% SYNTAX:
%   data = load_pipeline_data(cfg)
%   data = load_pipeline_data(cfg, 'OutlierMethod', method)
%   data = load_pipeline_data(cfg, 'Verbose', true)
%
% INPUTS:
%   cfg - Configuration structure from config.m containing:
%         * paths.data: Path to data files
%         * paths.qc: Path to QC results (for 'qc' method)
%         * paths.eda: Path to EDA results (for 'eda' method)
%
% OPTIONAL NAME-VALUE PAIRS:
%   'OutlierMethod' - Method for outlier detection. Options:
%                     'eda' (default): Use EDA T²-Q outlier detection
%                     'qc': Use legacy quality control flags
%                     'none': No outlier filtering
%   'Verbose'       - Display detailed output (default: true)
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
%          * pca_model: PCA model (if OutlierMethod='eda', otherwise empty)
%            - coeff: PCA coefficients
%            - n_comp: Number of components
%            - X_mean: Mean spectrum for centering
%            - explained: Variance explained
%
% EXAMPLES:
%   % Use EDA-based outlier detection (recommended)
%   cfg = config();
%   data = load_pipeline_data(cfg);
%
%   % Use legacy QC method
%   data = load_pipeline_data(cfg, 'OutlierMethod', 'qc');
%
%   % No outlier filtering
%   data = load_pipeline_data(cfg, 'OutlierMethod', 'none', 'Verbose', false);
%
% NOTES:
%   - EDA method requires run_eda.m to be executed first
%   - QC method requires quality_control_analysis.m to be executed first
%   - Patient_ID used for patient-wise stratification in cross-validation
%   - Diss_ID identifies individual samples/probes
%
% See also: config, run_eda, run_pipeline, run_patientwise_cv_direct

function data = load_pipeline_data(cfg, varargin)
    %% Parse Inputs
    p = inputParser;
    addRequired(p, 'cfg', @isstruct);
    addParameter(p, 'OutlierMethod', 'eda', @(x) ismember(x, {'eda', 'qc', 'none'}));
    addParameter(p, 'Verbose', true, @islogical);
    parse(p, cfg, varargin{:});
    
    method = p.Results.OutlierMethod;
    verbose = p.Results.Verbose;
    
    %% Validate cfg structure
    if ~isfield(cfg, 'paths')
        error('load_pipeline_data:InvalidConfig', ...
              'cfg must contain paths field');
    end
    
    %% Load Raw Data Tables
    if verbose
        fprintf('\n=== LOADING PIPELINE DATA ===\n');
        fprintf('Outlier method: %s\n', method);
        fprintf('Loading data tables...\n');
    end
    
    data_file_train = fullfile(cfg.paths.data, 'data_table_train.mat');
    data_file_test = fullfile(cfg.paths.data, 'data_table_test.mat');
    wavenumbers_file = fullfile(cfg.paths.data, 'wavenumbers.mat');
    
    % Check files exist
    if ~exist(data_file_train, 'file')
        error('load_pipeline_data:FileNotFound', ...
              'Training data file not found: %s', data_file_train);
    end
    if ~exist(data_file_test, 'file')
        error('load_pipeline_data:FileNotFound', ...
              'Test data file not found: %s', data_file_test);
    end
    
    load(data_file_train, 'data_table_train');
    load(data_file_test, 'data_table_test');
    load(wavenumbers_file, 'wavenumbers_roi');
    
    %% Load Outlier Information Based on Method
    switch method
        case 'eda'
            [outlier_info, pca_model] = load_eda_results(cfg, verbose);
        case 'qc'
            outlier_info = load_qc_results(cfg, verbose);
            pca_model = [];
        case 'none'
            outlier_info = create_no_filtering(data_table_train, data_table_test);
            pca_model = [];
            if verbose
                fprintf('No outlier filtering - using all spectra\n');
            end
    end
    
    %% Process Training Data
    if verbose
        fprintf('Processing training data...\n');
    end
    train = process_dataset(data_table_train, outlier_info.train, 'train', verbose);
    
    %% Process Test Data
    if verbose
        fprintf('Processing test data...\n');
    end
    test = process_dataset(data_table_test, outlier_info.test, 'test', verbose);
    
    %% Package Output
    data = struct();
    data.train = train;
    data.test = test;
    data.wavenumbers = wavenumbers_roi;
    data.pca_model = pca_model;
    data.outlier_method = method;
    
    %% Summary Statistics
    if verbose
        fprintf('\n=== DATA SUMMARY ===\n');
        fprintf('Training set:\n');
        fprintf('  Samples (Diss_IDs): %d\n', train.n_samples);
        fprintf('  Unique patients: %d\n', length(unique(train.patient_id)));
        fprintf('  WHO-1: %d samples, WHO-3: %d samples\n', ...
                sum(train.labels == 1), sum(train.labels == 3));
        fprintf('  Total spectra: %d\n', train.total_spectra);
        if isfield(train, 'n_spectra_removed')
            fprintf('  Spectra removed: %d (%.1f%%)\n', ...
                    train.n_spectra_removed, ...
                    100*train.n_spectra_removed/(train.total_spectra + train.n_spectra_removed));
        end
        
        fprintf('\nTest set:\n');
        fprintf('  Samples (Diss_IDs): %d\n', test.n_samples);
        fprintf('  Unique patients: %d\n', length(unique(test.patient_id)));
        fprintf('  WHO-1: %d samples, WHO-3: %d samples\n', ...
                sum(test.labels == 1), sum(test.labels == 3));
        fprintf('  Total spectra: %d\n', test.total_spectra);
        
        if ~isempty(pca_model)
            fprintf('\nPCA Model:\n');
            fprintf('  Components: %d PCs\n', pca_model.n_comp);
            fprintf('  Variance explained: %.2f%%\n', pca_model.total_variance);
        end
        fprintf('====================\n\n');
    end
end


%% Helper: Load EDA results
function [outlier_info, pca_model] = load_eda_results(cfg, verbose)
    eda_file = fullfile(cfg.paths.eda, 'eda_results_PP1.mat');
    
    if ~exist(eda_file, 'file')
        error('load_pipeline_data:EDANotFound', ...
              ['EDA results not found: %s\n' ...
               'Please run run_eda() first to generate EDA results.'], eda_file);
    end
    
    if verbose
        fprintf('Loading EDA results...\n');
    end
    load(eda_file, 'eda_results');
    
    % Extract outlier information
    outliers_both = eda_results.pca.outliers_both;
    pca_info = eda_results.pca;
    
    % Use ProbeUIDs if available
    if isfield(eda_results, 'probe_uids_pca')
        all_probe_ids_pca = eda_results.probe_uids_pca;
    else
        all_probe_ids_pca = eda_results.probe_ids_pca;
    end
    
    all_is_train = eda_results.is_train;
    
    if verbose
        fprintf('  EDA outliers detected: %d / %d training spectra (%.1f%%)\n', ...
                sum(outliers_both), length(outliers_both), ...
                100*sum(outliers_both)/length(outliers_both));
    end
    
    % Package outlier info
    outlier_info = struct();
    outlier_info.train.method = 'eda';
    outlier_info.train.outlier_flags = outliers_both;
    outlier_info.train.probe_ids_pca = all_probe_ids_pca;
    outlier_info.train.is_train = all_is_train;
    outlier_info.test.method = 'eda';
    outlier_info.test.outlier_flags = [];  % No filtering for test set
    
    % Package PCA model (first 15 components)
    n_pcs_to_use = min(15, size(pca_info.coeff, 2));
    pca_model = struct();
    pca_model.coeff = pca_info.coeff(:, 1:n_pcs_to_use);
    pca_model.n_comp = n_pcs_to_use;
    pca_model.explained = pca_info.explained(1:n_pcs_to_use);
    pca_model.X_mean = eda_results.X_mean;
    pca_model.total_variance = sum(pca_model.explained);
end


%% Helper: Load QC results
function outlier_info = load_qc_results(cfg, verbose)
    qc_file = fullfile(cfg.paths.qc, 'qc_flags.mat');
    
    if ~exist(qc_file, 'file')
        if verbose
            warning('load_pipeline_data:QCNotFound', ...
                    'QC results not found - using all spectra');
        end
        outlier_info = struct();
        outlier_info.train.method = 'none';
        outlier_info.test.method = 'none';
        return;
    end
    
    if verbose
        fprintf('Loading QC results...\n');
    end
    load(qc_file, 'qc_results');
    
    outlier_info = struct();
    outlier_info.train = qc_results.train;
    outlier_info.train.method = 'qc';
    outlier_info.test = qc_results.test;
    outlier_info.test.method = 'qc';
end


%% Helper: Create no-filtering structure
function outlier_info = create_no_filtering(dataTableTrain, dataTableTest)
    outlier_info = struct();
    
    % Training set - accept all
    outlier_info.train.method = 'none';
    outlier_info.train.sample_metrics = table();
    outlier_info.train.sample_metrics.Outlier_Flag = false(height(dataTableTrain), 1);
    outlier_info.train.valid_spectra_masks = cell(height(dataTableTrain), 1);
    for i = 1:height(dataTableTrain)
        n_spec = size(dataTableTrain.CombinedSpectra{i}, 1);
        outlier_info.train.valid_spectra_masks{i} = true(n_spec, 1);
    end
    
    % Test set - accept all
    outlier_info.test.method = 'none';
    outlier_info.test.sample_metrics = table();
    outlier_info.test.sample_metrics.Outlier_Flag = false(height(dataTableTest), 1);
    outlier_info.test.valid_spectra_masks = cell(height(dataTableTest), 1);
    for i = 1:height(dataTableTest)
        n_spec = size(dataTableTest.CombinedSpectra{i}, 1);
        outlier_info.test.valid_spectra_masks{i} = true(n_spec, 1);
    end
end


%% Helper: Process a dataset with outlier filtering
function dataset = process_dataset(dataTable, outlier_info, dataset_name, verbose)
    
    % Handle different outlier methods
    if strcmp(outlier_info.method, 'eda')
        dataset = process_dataset_eda(dataTable, outlier_info, dataset_name, verbose);
    elseif strcmp(outlier_info.method, 'qc')
        dataset = process_dataset_qc(dataTable, outlier_info, dataset_name, verbose);
    else  % 'none'
        dataset = process_dataset_qc(dataTable, outlier_info, dataset_name, verbose);
    end
end


%% Helper: Process dataset with EDA outlier filtering
function dataset = process_dataset_eda(dataTable, outlier_info, dataset_name, verbose)
    n_samples = height(dataTable);
    dataset = struct();
    dataset.diss_id = cell(n_samples, 1);
    dataset.patient_id = cell(n_samples, 1);
    dataset.spectra = cell(n_samples, 1);
    dataset.labels = zeros(n_samples, 1);
    dataset.metadata = struct();
    
    total_spectra = 0;
    n_spectra_removed = 0;
    sample_idx = 1;
    
    % Process each sample
    for i = 1:n_samples
        all_spectra = dataTable.CombinedSpectra{i};
        n_spec = size(all_spectra, 1);
        
        % Get ProbeUID for this sample
        probe_uid = dataTable.ProbeUID(i);
        
        % Find which spectra from this probe are in EDA data
        probe_mask = outlier_info.probe_ids_pca == probe_uid;
        
        if ~any(probe_mask)
            % Probe not in EDA data - keep all spectra
            valid_mask = true(n_spec, 1);
            if verbose
                warning('Probe %d not found in EDA training data - keeping all spectra', probe_uid);
            end
        elseif sum(probe_mask) ~= n_spec
            % Mismatch in spectrum count
            if verbose
                warning(['Probe %d: spectrum count mismatch (%d in table, %d in EDA)\n' ...
                         'Keeping all spectra for this probe.'], ...
                        probe_uid, n_spec, sum(probe_mask));
            end
            valid_mask = true(n_spec, 1);
        else
            % Get outlier flags for this probe's spectra
            outlier_flags_this_probe = outlier_info.outlier_flags(probe_mask);
            % Invert: outliers = false (remove), non-outliers = true (keep)
            valid_mask = ~outlier_flags_this_probe;
        end
        
        % Filter spectra
        filtered_spectra = all_spectra(valid_mask, :);
        n_removed = sum(~valid_mask);
        n_spectra_removed = n_spectra_removed + n_removed;
        
        if isempty(filtered_spectra)
            if verbose
                warning('Sample %d has no valid spectra after EDA filtering - skipping', i);
            end
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
    
    % Trim arrays
    dataset.diss_id = dataset.diss_id(1:sample_idx-1);
    dataset.patient_id = dataset.patient_id(1:sample_idx-1);
    dataset.spectra = dataset.spectra(1:sample_idx-1);
    dataset.labels = dataset.labels(1:sample_idx-1);
    dataset.metadata.age = dataset.metadata.age(1:sample_idx-1);
    dataset.metadata.sex = dataset.metadata.sex(1:sample_idx-1);
    
    dataset.n_samples = sample_idx - 1;
    dataset.total_spectra = total_spectra;
    dataset.n_spectra_removed = n_spectra_removed;
end


%% Helper: Process dataset with QC filtering or no filtering
function dataset = process_dataset_qc(dataTable, outlier_info, dataset_name, verbose)
    % Apply QC or no filtering
    valid_samples = ~outlier_info.sample_metrics.Outlier_Flag;
    
    % Initialize output structure
    dataset = struct();
    dataset.n_samples = sum(valid_samples);
    dataset.diss_id = cell(dataset.n_samples, 1);
    dataset.patient_id = cell(dataset.n_samples, 1);
    dataset.spectra = cell(dataset.n_samples, 1);
    dataset.labels = zeros(dataset.n_samples, 1);
    dataset.metadata = struct();
    
    % Extract data from valid samples
    sample_idx = 1;
    total_spectra = 0;
    
    for i = 1:height(dataTable)
        if valid_samples(i)
            % Get QC-filtered spectra
            all_spectra = dataTable.CombinedSpectra{i};
            valid_mask = outlier_info.valid_spectra_masks{i};
            filtered_spectra = all_spectra(valid_mask, :);
            
            if isempty(filtered_spectra)
                if verbose
                    warning('Sample %d has no valid spectra - skipping', i);
                end
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
    
    % Trim to actual size
    dataset.n_samples = sample_idx - 1;
    dataset.diss_id = dataset.diss_id(1:dataset.n_samples);
    dataset.patient_id = dataset.patient_id(1:dataset.n_samples);
    dataset.spectra = dataset.spectra(1:dataset.n_samples);
    dataset.labels = dataset.labels(1:dataset.n_samples);
    dataset.metadata.age = dataset.metadata.age(1:dataset.n_samples);
    dataset.metadata.sex = dataset.metadata.sex(1:dataset.n_samples);
    
    dataset.total_spectra = total_spectra;
end
