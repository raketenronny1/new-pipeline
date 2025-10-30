classdef DataLoader < handle
    %DATALOADER Flexible data loader for FTIR spectroscopy classification
    %
    % DESCRIPTION:
    %   Loads and validates spectral data from various sources with automatic
    %   format detection and comprehensive validation. Supports both complete
    %   datasets and pre-split train/test sets.
    %
    % USAGE:
    %   % Load complete dataset
    %   [X, y, pids, meta] = DataLoader.load('data/dataset_complete.mat');
    %   
    %   % Load train/test split
    %   [data_train, data_test, meta] = DataLoader.loadSplit('data/');
    %   
    %   % Load with specific preprocessing variant
    %   [X, y, pids, meta] = DataLoader.load('data/data_table_train.mat', ...
    %                                         'SpectraField', 'CombinedSpectra_PP1');
    %
    % DATA FORMAT:
    %   Expects MATLAB table with columns:
    %   - Patient_ID: Unique patient identifier
    %   - WHO_Grade: Class labels (1 or 3)
    %   - CombinedSpectra_PP1/PP2/Raw: Cell arrays of spectral matrices
    %   - Diss_ID: Sample/probe identifiers
    %
    % OUTPUTS:
    %   X: [n_samples × n_features] spectral matrix (mean of spectra per sample)
    %   y: Categorical labels (WHO grade)
    %   pids: Patient IDs for each sample
    %   meta: Metadata structure with dataset statistics
    %
    % See also: Config, PreprocessingPipeline
    
    methods (Static)
        function [X, y, patient_ids, metadata] = load(filepath, varargin)
            %LOAD Load spectral data from file with auto-detection
            %
            % SYNTAX:
            %   [X, y, pids, meta] = DataLoader.load(filepath)
            %   [X, y, pids, meta] = DataLoader.load(filepath, 'SpectraField', 'CombinedSpectra_PP1')
            %   [X, y, pids, meta] = DataLoader.load(filepath, 'AggregationMethod', 'median')
            %
            % NAME-VALUE PAIRS:
            %   SpectraField      - Which spectra field to use (default: auto-detect)
            %   AggregationMethod - 'mean' or 'median' for multi-spectra samples (default: 'mean')
            %   HandleNaN         - 'error', 'remove', or 'impute' (default: 'error')
            %   Verbose           - Display progress (default: true)
            
            % Parse inputs
            p = inputParser;
            addRequired(p, 'filepath', @(x) ischar(x) || isstring(x));
            addParameter(p, 'SpectraField', 'auto', @(x) ischar(x) || isstring(x));
            addParameter(p, 'AggregationMethod', 'mean', @(x) ismember(x, {'mean', 'median'}));
            addParameter(p, 'HandleNaN', 'error', @(x) ismember(x, {'error', 'remove', 'impute'}));
            addParameter(p, 'Verbose', true, @islogical);
            parse(p, filepath, varargin{:});
            
            spectra_field = p.Results.SpectraField;
            agg_method = p.Results.AggregationMethod;
            handle_nan = p.Results.HandleNaN;
            verbose = p.Results.Verbose;
            
            if verbose
                fprintf('=== DataLoader: Loading %s ===\n', filepath);
            end
            
            % Load file
            [data_table, format] = DataLoader.loadFile(filepath, verbose);
            
            % Auto-detect spectra field if needed
            if strcmp(spectra_field, 'auto')
                spectra_field = DataLoader.detectSpectraField(data_table, verbose);
            end
            
            % Extract data
            [X, y, patient_ids] = DataLoader.extractData(data_table, spectra_field, ...
                                                         agg_method, verbose);
            
            % Handle NaN values
            DataLoader.handleNaN(X, handle_nan);
            
            % Validate data
            DataLoader.validateData(X, y, patient_ids);
            
            % Compute metadata
            metadata = DataLoader.computeMetadata(X, y, patient_ids, data_table, verbose);
            
            if verbose
                fprintf('✓ Data loaded successfully\n');
                fprintf('  %d samples, %d features, %d patients\n', ...
                    metadata.n_samples, metadata.n_features, metadata.n_patients);
            end
        end
        
        function [data_train, data_test, metadata] = loadSplit(data_dir, varargin)
            %LOADSPLIT Load pre-split train/test datasets
            %
            % SYNTAX:
            %   [train, test, meta] = DataLoader.loadSplit('data/')
            %   [train, test, meta] = DataLoader.loadSplit('data/', 'SpectraField', 'CombinedSpectra_PP2')
            
            % Parse inputs
            p = inputParser;
            addRequired(p, 'data_dir', @(x) ischar(x) || isstring(x));
            addParameter(p, 'SpectraField', 'auto', @(x) ischar(x) || isstring(x));
            addParameter(p, 'AggregationMethod', 'mean', @(x) ismember(x, {'mean', 'median'}));
            addParameter(p, 'Verbose', true, @islogical);
            parse(p, data_dir, varargin{:});
            
            spectra_field = p.Results.SpectraField;
            agg_method = p.Results.AggregationMethod;
            verbose = p.Results.Verbose;
            
            % Construct file paths
            train_file = fullfile(data_dir, 'data_table_train.mat');
            test_file = fullfile(data_dir, 'data_table_test.mat');
            wavenumbers_file = fullfile(data_dir, 'wavenumbers.mat');
            split_info_file = fullfile(data_dir, 'split_info.mat');
            
            % Load training data
            [X_train, y_train, pids_train, meta_train] = ...
                DataLoader.load(train_file, 'SpectraField', spectra_field, ...
                               'AggregationMethod', agg_method, 'Verbose', verbose);
            
            % Load test data
            [X_test, y_test, pids_test, meta_test] = ...
                DataLoader.load(test_file, 'SpectraField', spectra_field, ...
                               'AggregationMethod', agg_method, 'Verbose', verbose);
            
            % Package training data
            data_train = struct();
            data_train.X = X_train;
            data_train.y = y_train;
            data_train.patient_ids = pids_train;
            data_train.metadata = meta_train;
            
            % Package test data
            data_test = struct();
            data_test.X = X_test;
            data_test.y = y_test;
            data_test.patient_ids = pids_test;
            data_test.metadata = meta_test;
            
            % Load wavenumbers if available
            if exist(wavenumbers_file, 'file')
                loaded = load(wavenumbers_file);
                if isfield(loaded, 'wavenumbers_roi')
                    data_train.wavenumbers = loaded.wavenumbers_roi;
                    data_test.wavenumbers = loaded.wavenumbers_roi;
                end
            end
            
            % Load split info if available
            if exist(split_info_file, 'file')
                loaded = load(split_info_file);
                if isfield(loaded, 'split_info')
                    metadata.split_info = loaded.split_info;
                else
                    metadata.split_info = [];
                end
            else
                metadata.split_info = [];
            end
            
            % Validate no patient overlap
            overlap = intersect(unique(pids_train), unique(pids_test));
            if ~isempty(overlap)
                error('DataLoader:PatientLeakage', ...
                    'Patient overlap detected between train and test sets!');
            end
            
            % Combined metadata
            metadata.train = meta_train;
            metadata.test = meta_test;
            metadata.total_patients = meta_train.n_patients + meta_test.n_patients;
            metadata.total_samples = meta_train.n_samples + meta_test.n_samples;
            
            if verbose
                fprintf('\n=== Split Summary ===\n');
                fprintf('Train: %d samples, %d patients\n', ...
                    meta_train.n_samples, meta_train.n_patients);
                fprintf('Test:  %d samples, %d patients\n', ...
                    meta_test.n_samples, meta_test.n_patients);
                fprintf('✓ No patient overlap verified\n');
            end
        end
    end
    
    methods (Static, Access = private)
        function [data_table, format] = loadFile(filepath, verbose)
            %LOADFILE Load data file with format detection
            
            % Check file exists
            if ~exist(filepath, 'file')
                error('DataLoader:FileNotFound', 'File not found: %s', filepath);
            end
            
            [~, ~, ext] = fileparts(filepath);
            
            switch lower(ext)
                case '.mat'
                    % Load MAT file
                    loaded = load(filepath);
                    
                    % Auto-detect main variable
                    vars = fieldnames(loaded);
                    
                    % Try common names first
                    possible_names = {'data_table_train', 'data_table_test', ...
                                     'dataset_men', 'dataset_complete', 'data'};
                    
                    data_table = [];
                    for i = 1:length(possible_names)
                        if isfield(loaded, possible_names{i})
                            data_table = loaded.(possible_names{i});
                            break;
                        end
                    end
                    
                    % If not found, use first table variable
                    if isempty(data_table)
                        for i = 1:length(vars)
                            if istable(loaded.(vars{i}))
                                data_table = loaded.(vars{i});
                                break;
                            end
                        end
                    end
                    
                    if isempty(data_table)
                        error('DataLoader:NoTable', 'No table found in MAT file');
                    end
                    
                    format = 'mat';
                    
                otherwise
                    error('DataLoader:UnsupportedFormat', ...
                        'Unsupported file format: %s', ext);
            end
            
            if verbose
                fprintf('  Loaded %s (%d rows)\n', filepath, height(data_table));
            end
        end
        
        function spectra_field = detectSpectraField(data_table, verbose)
            %DETECTSPECTRAFIELD Auto-detect which spectra field to use
            
            vars = data_table.Properties.VariableNames;
            
            % Look for RawSpectra field first (contains all spectra)
            if ismember('RawSpectra', vars)
                spectra_field = 'RawSpectra';
                if verbose
                    fprintf('  Using spectra field: %s\n', spectra_field);
                end
                return;
            end
            
            % Second priority: MeanRawSpectrum (already aggregated)
            if ismember('MeanRawSpectrum', vars)
                spectra_field = 'MeanRawSpectrum';
                if verbose
                    fprintf('  Using spectra field: %s (already aggregated per sample)\n', spectra_field);
                end
                warning('DataLoader:UsingMeanSpectrum', ...
                    'Using MeanRawSpectrum - already aggregated per sample');
                return;
            end
            
            % Fallback: Check for old CombinedRawSpectra (data not cleaned)
            if ismember('CombinedRawSpectra', vars)
                spectra_field = 'CombinedRawSpectra';
                if verbose
                    fprintf('  Using spectra field: %s (DEPRECATED)\n', spectra_field);
                end
                warning('DataLoader:Uncleaned', ...
                    'Using CombinedRawSpectra - data may not be cleaned. Run prepare_data.m first!');
                return;
            end
            
            error('DataLoader:NoSpectraField', [...
                'No raw spectra field found! Expected RawSpectra or MeanRawSpectrum. ' ...
                'Run prepare_data.m first to clean data files.']);
        end
        
        function [X, y, patient_ids] = extractData(data_table, spectra_field, agg_method, verbose)
            %EXTRACTDATA Extract and aggregate spectral data
            
            % Get required fields
            if ~ismember(spectra_field, data_table.Properties.VariableNames)
                error('DataLoader:InvalidField', ...
                    'Field %s not found in data table', spectra_field);
            end
            
            spectra_cell = data_table.(spectra_field);
            
            % Extract labels (WHO_Grade)
            if ismember('WHO_Grade', data_table.Properties.VariableNames)
                labels = data_table.WHO_Grade;
            elseif ismember('label', data_table.Properties.VariableNames)
                labels = data_table.label;
            else
                error('DataLoader:NoLabels', 'No label field found in table');
            end
            
            % Extract patient IDs
            if ismember('Patient_ID', data_table.Properties.VariableNames)
                patient_ids = data_table.Patient_ID;
            elseif ismember('patient', data_table.Properties.VariableNames)
                patient_ids = data_table.patient;
            else
                error('DataLoader:NoPatientID', 'No patient ID field found in table');
            end
            
            % Aggregate spectra (each cell may contain multiple spectra)
            n_samples = length(spectra_cell);
            n_features = size(spectra_cell{1}, 2);  % Assume all same size
            
            X = zeros(n_samples, n_features);
            
            if verbose
                fprintf('  Aggregating spectra using %s...\n', agg_method);
            end
            
            for i = 1:n_samples
                spectra = spectra_cell{i};
                
                if isempty(spectra)
                    error('DataLoader:EmptySpectra', 'Empty spectra at sample %d', i);
                end
                
                % Aggregate multiple spectra per sample
                switch agg_method
                    case 'mean'
                        X(i, :) = mean(spectra, 1);
                    case 'median'
                        X(i, :) = median(spectra, 1);
                end
            end
            
            % Convert labels to categorical
            if isnumeric(labels)
                y = categorical(labels);
            elseif iscategorical(labels)
                y = labels;
            else
                y = categorical(labels);
            end
        end
        
        function handleNaN(X, method)
            %HANDLENAN Handle NaN values in data
            
            if ~any(isnan(X(:)))
                return;
            end
            
            switch method
                case 'error'
                    error('DataLoader:NaNDetected', ...
                        'NaN values detected in data. Use HandleNaN option to handle.');
                    
                case 'remove'
                    warning('DataLoader:NaNRemoved', ...
                        'Removing %d samples with NaN values', ...
                        sum(any(isnan(X), 2)));
                    % This would require returning modified X, y, pids
                    % Implementation left for future if needed
                    error('DataLoader:NotImplemented', ...
                        'NaN removal not yet implemented');
                    
                case 'impute'
                    warning('DataLoader:NaNImputed', 'Imputing NaN values with column means');
                    for j = 1:size(X, 2)
                        col = X(:, j);
                        col(isnan(col)) = mean(col(~isnan(col)));
                        X(:, j) = col;
                    end
            end
        end
        
        function validateData(X, y, patient_ids)
            %VALIDATEDATA Validate extracted data
            
            % Type validation
            validateattributes(X, {'double'}, {'2d', 'finite'}, ...
                'DataLoader', 'X');
            validateattributes(y, {'categorical'}, {'vector'}, ...
                'DataLoader', 'y');
            
            % Dimension validation
            assert(size(X, 1) == length(y), ...
                'DataLoader:DimensionMismatch', ...
                'Number of samples (%d) does not match labels (%d)', ...
                size(X, 1), length(y));
            
            assert(size(X, 1) == length(patient_ids), ...
                'DataLoader:DimensionMismatch', ...
                'Number of samples (%d) does not match patient IDs (%d)', ...
                size(X, 1), length(patient_ids));
            
            % Class validation
            assert(length(unique(y)) >= 2, ...
                'DataLoader:InsufficientClasses', ...
                'Data must contain at least 2 classes');
            
            % Feature validation
            assert(size(X, 2) >= 10, ...
                'DataLoader:InsufficientFeatures', ...
                'Data must contain at least 10 features');
        end
        
        function metadata = computeMetadata(X, y, patient_ids, data_table, verbose)
            %COMPUTEMETADATA Compute dataset metadata and statistics
            
            metadata = struct();
            
            % Basic counts
            metadata.n_samples = size(X, 1);
            metadata.n_features = size(X, 2);
            metadata.n_patients = length(unique(patient_ids));
            
            % Class distribution
            [classes, ~, ic] = unique(y);
            class_counts = histcounts(ic, 1:length(classes)+1);
            metadata.classes = classes;
            metadata.class_counts = class_counts;
            metadata.class_distribution = array2table([cellstr(classes), ...
                num2cell(class_counts')], ...
                'VariableNames', {'Class', 'Count'});
            
            % Samples per patient
            [~, ~, ic_patient] = unique(patient_ids);
            patient_counts = histcounts(ic_patient, 1:metadata.n_patients+1);
            metadata.samples_per_patient.mean = mean(patient_counts);
            metadata.samples_per_patient.std = std(patient_counts);
            metadata.samples_per_patient.min = min(patient_counts);
            metadata.samples_per_patient.max = max(patient_counts);
            
            % Feature ranges
            metadata.feature_range.min = min(X, [], 1);
            metadata.feature_range.max = max(X, [], 1);
            metadata.feature_range.mean = mean(X, 1);
            metadata.feature_range.std = std(X, 0, 1);
            
            % Additional table info if available
            if nargin >= 4 && istable(data_table)
                vars = data_table.Properties.VariableNames;
                metadata.available_fields = vars;
            end
            
            if verbose
                fprintf('\n  --- Metadata ---\n');
                fprintf('  Classes: %s\n', strjoin(cellstr(classes), ', '));
                fprintf('  Samples per patient: %.1f ± %.1f (range: %d-%d)\n', ...
                    metadata.samples_per_patient.mean, ...
                    metadata.samples_per_patient.std, ...
                    metadata.samples_per_patient.min, ...
                    metadata.samples_per_patient.max);
            end
        end
    end
end
