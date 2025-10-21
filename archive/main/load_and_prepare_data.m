
%% PHASE 1: DATA LOADING & INTEGRATION WITH QC
% This script loads the raw data and integrates it with QC results to create
% analysis-ready datasets.

function load_and_prepare_data(cfg)
    %% Load Raw Data
    fprintf('Loading raw data...\n');
    load(fullfile(cfg.paths.data, 'data_table_train.mat'), 'dataTableTrain');
    load(fullfile(cfg.paths.data, 'data_table_test.mat'), 'dataTableTest');
    load(fullfile(cfg.paths.data, 'wavenumbers.mat'), 'wavenumbers_roi');

    %% Load QC Results
    fprintf('Loading QC results...\n');
    load(fullfile(cfg.paths.qc, 'qc_flags.mat'), 'qc_results');

%% Process Training Set
fprintf('Processing training set...\n');

% Get samples that passed QC
samples_to_keep_train = ~qc_results.train.sample_metrics.Outlier_Flag;

% Calculate representative spectra for retained samples
n_train_samples = sum(samples_to_keep_train);
n_wavenumbers = length(wavenumbers_roi);
X_train = zeros(n_train_samples, n_wavenumbers);

valid_sample_idx = 1;
for i = 1:height(dataTableTrain)
    if samples_to_keep_train(i)
        % Get valid spectra for this sample
        valid_spectra = dataTableTrain.CombinedSpectra{i}(qc_results.train.valid_spectra_masks{i}, :);
        
        % Calculate representative spectrum (quality-filtered mean)
        if ~isempty(valid_spectra)
            X_train(valid_sample_idx, :) = mean(valid_spectra, 1, 'omitnan');
            
            % Check for NaN/Inf after mean calculation
            if any(isnan(X_train(valid_sample_idx, :)) | isinf(X_train(valid_sample_idx, :)))
                warning('Sample %d has invalid values after averaging', i);
                % Use median instead as fallback
                X_train(valid_sample_idx, :) = median(valid_spectra, 1, 'omitnan');
            end
            
            valid_sample_idx = valid_sample_idx + 1;
        else
            warning('No valid spectra for training sample %d', i);
        end
    end
end

% Extract corresponding labels and metadata
y_train = dataTableTrain.WHO_Grade(samples_to_keep_train);
probe_ids_train = dataTableTrain.Diss_ID(samples_to_keep_train);

%% Process Test Set
fprintf('Processing test set...\n');

% Get samples that passed QC
samples_to_keep_test = ~qc_results.test.sample_metrics.Outlier_Flag;

% Calculate representative spectra for retained samples
n_test_samples = sum(samples_to_keep_test);
X_test = zeros(n_test_samples, n_wavenumbers);

valid_sample_idx = 1;
for i = 1:height(dataTableTest)
    if samples_to_keep_test(i)
        % Get valid spectra for this sample
        valid_spectra = dataTableTest.CombinedSpectra{i}(qc_results.test.valid_spectra_masks{i}, :);
        
        % Calculate representative spectrum
        if ~isempty(valid_spectra)
            X_test(valid_sample_idx, :) = mean(valid_spectra, 1, 'omitnan');
            
            % Check for NaN/Inf after mean calculation
            if any(isnan(X_test(valid_sample_idx, :)) | isinf(X_test(valid_sample_idx, :)))
                warning('Sample %d has invalid values after averaging', i);
                % Use median instead as fallback
                X_test(valid_sample_idx, :) = median(valid_spectra, 1, 'omitnan');
            end
            
            valid_sample_idx = valid_sample_idx + 1;
        else
            warning('No valid spectra for test sample %d', i);
        end
    end
end

% Extract corresponding labels and metadata
y_test = dataTableTest.WHO_Grade(samples_to_keep_test);
probe_ids_test = dataTableTest.Diss_ID(samples_to_keep_test);

%% Create Analysis-Ready Datasets
fprintf('Creating analysis-ready datasets...\n');

% Training data structure
trainingData = struct();
trainingData.X = X_train;

% Debug y_train type and values
fprintf('DEBUG: y_train class: %s\n', class(y_train));
if iscategorical(y_train)
    fprintf('DEBUG: y_train is categorical with %d unique values\n', numel(categories(y_train)));
    fprintf('DEBUG: y_train categories: %s\n', strjoin(string(categories(y_train)), ', '));
else
    fprintf('DEBUG: First few y_train values: ');
    if iscell(y_train) || isstring(y_train)
        for i = 1:min(5, numel(y_train))
            fprintf('%s ', string(y_train(i)));
        end
    else
        for i = 1:min(5, numel(y_train))
            fprintf('%d ', y_train(i));
        end
    end
    fprintf('\n');
end

% Ensure y_train is numeric for model training
if iscategorical(y_train)
    trainingData.y = double(y_train);
else
    trainingData.y = y_train;
end

% Debug trainingData.y after setting
fprintf('DEBUG: trainingData.y field type: %s\n', class(trainingData.y));
if isfield(trainingData, 'y') 
    fprintf('DEBUG: trainingData has y field\n');
else
    fprintf('DEBUG: ERROR - trainingData does not have y field!\n');
end
trainingData.probe_ids = probe_ids_train;
trainingData.probe_table = dataTableTrain(samples_to_keep_train, :);

% Test data structure
testData = struct();
testData.X = X_test;
% Ensure y_test is numeric for model evaluation
if iscategorical(y_test)
    testData.y = double(y_test);
else
    testData.y = y_test;
end
testData.probe_ids = probe_ids_test;
testData.probe_table = dataTableTest(samples_to_keep_test, :);

%% Final Quality Checks
% Verify dimensions
fprintf('\nDataset dimensions after QC:\n');
fprintf('Training set: %d samples × %d wavenumbers\n', size(X_train));
fprintf('Test set: %d samples × %d wavenumbers\n', size(X_test));

% Check class balance
fprintf('\nClass distribution:\n');
% Handle categorical variables
if iscategorical(y_train)
    train_who1 = sum(double(y_train)==1);
    train_who3 = sum(double(y_train)==3);
else
    train_who1 = sum(y_train==1);
    train_who3 = sum(y_train==3);
end
fprintf('Training WHO-1: %d, WHO-3: %d\n', train_who1, train_who3);

% Handle categorical variables
if iscategorical(y_test)
    test_who1 = sum(double(y_test)==1);
    test_who3 = sum(double(y_test)==3);
else
    test_who1 = sum(y_test==1);
    test_who3 = sum(y_test==3);
end
fprintf('Test WHO-1: %d, WHO-3: %d\n', test_who1, test_who3);

% Verify data validity
nan_locs_train = find(isnan(X_train(:)));
inf_locs_train = find(isinf(X_train(:)));
nan_locs_test = find(isnan(X_test(:)));
inf_locs_test = find(isinf(X_test(:)));

if ~isempty(nan_locs_train)
    [rows, cols] = ind2sub(size(X_train), nan_locs_train);
    error('Training data contains NaN values at samples: %s', num2str(unique(rows)));
end

if ~isempty(inf_locs_train)
    [rows, cols] = ind2sub(size(X_train), inf_locs_train);
    error('Training data contains Inf values at samples: %s', num2str(unique(rows)));
end

if ~isempty(nan_locs_test)
    [rows, cols] = ind2sub(size(X_test), nan_locs_test);
    error('Test data contains NaN values at samples: %s', num2str(unique(rows)));
end

if ~isempty(inf_locs_test)
    [rows, cols] = ind2sub(size(X_test), inf_locs_test);
    error('Test data contains Inf values at samples: %s', num2str(unique(rows)));
end

    %% Save Processed Data
    fprintf('Saving processed data...\n');
    save(fullfile(cfg.paths.results, 'preprocessed_data.mat'), 'trainingData', 'testData', 'wavenumbers_roi');

    fprintf('Data preparation complete.\n');
end