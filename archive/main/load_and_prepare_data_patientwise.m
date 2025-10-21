%% PATIENT-WISE DATA LOADING (NO AVERAGING)
% This script loads the raw data and integrates it with QC results to create
% patient-indexed data structures WITHOUT averaging spectra.
%
% Following best practices from:
% - Baker et al. (2014) Nature Protocols 9(8):1771-1791
% - Greener et al. (2022) Nature Reviews Molecular Cell Biology 23:40-55
%
% Key principles:
% - Patient-wise organization (no data leakage)
% - NO spectrum averaging before prediction
% - All ~768 spectra per patient preserved individually
% - Quality control applied at spectrum level

function load_and_prepare_data_patientwise(cfg)
    %% Load Raw Data
    fprintf('Loading raw data...\n');
    load(fullfile(cfg.paths.data, 'data_table_train.mat'), 'dataTableTrain');
    load(fullfile(cfg.paths.data, 'data_table_test.mat'), 'dataTableTest');
    load(fullfile(cfg.paths.data, 'wavenumbers.mat'), 'wavenumbers_roi');

    %% Load QC Results
    fprintf('Loading QC results...\n');
    qc_file = fullfile(cfg.paths.qc, 'qc_flags.mat');
    if exist(qc_file, 'file')
        load(qc_file, 'qc_results');
    else
        warning('QC results not found. All spectra will be included.');
        % Create dummy QC structure that accepts all spectra
        qc_results = struct();
        qc_results.train.sample_metrics.Outlier_Flag = false(height(dataTableTrain), 1);
        qc_results.train.valid_spectra_masks = cell(height(dataTableTrain), 1);
        for i = 1:height(dataTableTrain)
            n_spectra = size(dataTableTrain.CombinedSpectra{i}, 1);
            qc_results.train.valid_spectra_masks{i} = true(n_spectra, 1);
        end
        qc_results.test.sample_metrics.Outlier_Flag = false(height(dataTableTest), 1);
        qc_results.test.valid_spectra_masks = cell(height(dataTableTest), 1);
        for i = 1:height(dataTableTest)
            n_spectra = size(dataTableTest.CombinedSpectra{i}, 1);
            qc_results.test.valid_spectra_masks{i} = true(n_spectra, 1);
        end
    end

    %% Process Training Set - PATIENT-WISE STRUCTURE
    fprintf('Creating patient-wise training data structure...\n');
    
    samples_to_keep_train = ~qc_results.train.sample_metrics.Outlier_Flag;
    n_patients_train = sum(samples_to_keep_train);
    
    % Initialize patient data structure
    patientDataTrain = struct('patientID', {}, ...
                              'spectra', {}, ...
                              'label', {}, ...
                              'probe_id', {}, ...
                              'metadata', {});
    
    patient_idx = 1;
    for i = 1:height(dataTableTrain)
        if samples_to_keep_train(i)
            % Get valid spectra for this patient (QC filtered)
            valid_spectra = dataTableTrain.CombinedSpectra{i}(qc_results.train.valid_spectra_masks{i}, :);
            
            if ~isempty(valid_spectra) && size(valid_spectra, 1) > 0
                % Store ALL spectra for this patient (no averaging!)
                % Use Diss_ID (probe/dissection ID) as unique identifier
                % NOTE: Same Patient_ID may have multiple Diss_IDs (recidival tumors)
                patientDataTrain(patient_idx).patientID = dataTableTrain.Diss_ID{i};  % Unique probe ID
                patientDataTrain(patient_idx).spectra = valid_spectra;  % [N_spectra × N_wavenumbers]
                
                % Extract label
                label_val = dataTableTrain.WHO_Grade(i);
                if iscategorical(label_val)
                    label_str = char(label_val);
                    if contains(label_str, '1')
                        patientDataTrain(patient_idx).label = 1;
                    elseif contains(label_str, '3')
                        patientDataTrain(patient_idx).label = 3;
                    else
                        error('Unexpected WHO grade: %s', label_str);
                    end
                else
                    patientDataTrain(patient_idx).label = double(label_val);
                end
                
                % Store additional metadata
                patientDataTrain(patient_idx).probe_id = dataTableTrain.Diss_ID{i};
                patientDataTrain(patient_idx).metadata = struct(...
                    'patient_id', dataTableTrain.Patient_ID{i}, ...  % Biological patient ID
                    'age', dataTableTrain.Age(i), ...
                    'sex', dataTableTrain.Sex(i), ...
                    'n_spectra', size(valid_spectra, 1), ...
                    'n_positions', dataTableTrain.NumPositions(i));
                
                patient_idx = patient_idx + 1;
            else
                warning('Patient %s has no valid spectra after QC', dataTableTrain.Diss_ID{i});
            end
        end
    end
    
    fprintf('Training set: %d patients loaded\n', length(patientDataTrain));
    
    %% Process Test Set - PATIENT-WISE STRUCTURE
    fprintf('Creating patient-wise test data structure...\n');
    
    samples_to_keep_test = ~qc_results.test.sample_metrics.Outlier_Flag;
    n_patients_test = sum(samples_to_keep_test);
    
    % Initialize patient data structure
    patientDataTest = struct('patientID', {}, ...
                             'spectra', {}, ...
                             'label', {}, ...
                             'probe_id', {}, ...
                             'metadata', {});
    
    patient_idx = 1;
    for i = 1:height(dataTableTest)
        if samples_to_keep_test(i)
            % Get valid spectra for this patient (QC filtered)
            valid_spectra = dataTableTest.CombinedSpectra{i}(qc_results.test.valid_spectra_masks{i}, :);
            
            if ~isempty(valid_spectra) && size(valid_spectra, 1) > 0
                % Store ALL spectra for this patient (no averaging!)
                % Use Diss_ID (probe/dissection ID) as unique identifier
                patientDataTest(patient_idx).patientID = dataTableTest.Diss_ID{i};  % Unique probe ID
                patientDataTest(patient_idx).spectra = valid_spectra;  % [N_spectra × N_wavenumbers]
                
                % Extract label
                label_val = dataTableTest.WHO_Grade(i);
                if iscategorical(label_val)
                    label_str = char(label_val);
                    if contains(label_str, '1')
                        patientDataTest(patient_idx).label = 1;
                    elseif contains(label_str, '3')
                        patientDataTest(patient_idx).label = 3;
                    else
                        error('Unexpected WHO grade: %s', label_str);
                    end
                else
                    patientDataTest(patient_idx).label = double(label_val);
                end
                
                % Store additional metadata
                patientDataTest(patient_idx).probe_id = dataTableTest.Diss_ID{i};
                patientDataTest(patient_idx).metadata = struct(...
                    'patient_id', dataTableTest.Patient_ID{i}, ...  % Biological patient ID
                    'age', dataTableTest.Age(i), ...
                    'sex', dataTableTest.Sex(i), ...
                    'n_spectra', size(valid_spectra, 1), ...
                    'n_positions', dataTableTest.NumPositions(i));
                
                patient_idx = patient_idx + 1;
            else
                warning('Patient %s has no valid spectra after QC', dataTableTest.Diss_ID{i});
            end
        end
    end
    
    fprintf('Test set: %d patients loaded\n', length(patientDataTest));
    
    %% Validate Data Structures
    fprintf('\nValidating patient data structures...\n');
    validatePatientData(patientDataTrain, 'Training');
    validatePatientData(patientDataTest, 'Test');
    
    %% Create Analysis-Ready Datasets
    fprintf('\nCreating analysis-ready datasets...\n');
    
    % Training data structure (patient-wise)
    trainingData = struct();
    trainingData.patientData = patientDataTrain;
    trainingData.wavenumbers = wavenumbers_roi;
    trainingData.n_patients = length(patientDataTrain);
    trainingData.n_wavenumbers = length(wavenumbers_roi);
    
    % Test data structure (patient-wise)
    testData = struct();
    testData.patientData = patientDataTest;
    testData.wavenumbers = wavenumbers_roi;
    testData.n_patients = length(patientDataTest);
    testData.n_wavenumbers = length(wavenumbers_roi);
    
    %% Summary Statistics
    fprintf('\n=== DATA SUMMARY ===\n');
    fprintf('Training set: %d patients\n', trainingData.n_patients);
    fprintf('Test set: %d patients\n', testData.n_patients);
    
    % Count total spectra
    total_spectra_train = sum(arrayfun(@(x) size(x.spectra, 1), patientDataTrain));
    total_spectra_test = sum(arrayfun(@(x) size(x.spectra, 1), patientDataTest));
    fprintf('Total spectra - Train: %d, Test: %d\n', total_spectra_train, total_spectra_test);
    
    % Class distribution (patient-level)
    labels_train = [patientDataTrain.label];
    labels_test = [patientDataTest.label];
    fprintf('\nClass distribution (PATIENT-LEVEL):\n');
    fprintf('Training: WHO-1: %d patients, WHO-3: %d patients\n', ...
            sum(labels_train == 1), sum(labels_train == 3));
    fprintf('Test: WHO-1: %d patients, WHO-3: %d patients\n', ...
            sum(labels_test == 1), sum(labels_test == 3));
    
    % Spectra per patient statistics
    spectra_counts_train = arrayfun(@(x) size(x.spectra, 1), patientDataTrain);
    spectra_counts_test = arrayfun(@(x) size(x.spectra, 1), patientDataTest);
    fprintf('\nSpectra per patient:\n');
    fprintf('Training: mean=%.0f, median=%.0f, min=%d, max=%d\n', ...
            mean(spectra_counts_train), median(spectra_counts_train), ...
            min(spectra_counts_train), max(spectra_counts_train));
    fprintf('Test: mean=%.0f, median=%.0f, min=%d, max=%d\n', ...
            mean(spectra_counts_test), median(spectra_counts_test), ...
            min(spectra_counts_test), max(spectra_counts_test));
    
    %% Save Processed Data
    fprintf('\nSaving patient-wise data structures...\n');
    save(fullfile(cfg.paths.results, 'patientwise_data.mat'), ...
         'trainingData', 'testData', 'wavenumbers_roi', '-v7.3');
    
    fprintf('✓ Patient-wise data preparation complete.\n');
    fprintf('  NO AVERAGING PERFORMED - All spectra preserved for individual prediction\n');
end


%% VALIDATION FUNCTION
function validatePatientData(patientData, dataset_name)
    % Validates patient data structure for integrity and consistency
    %
    % NOTE: patientID field contains Diss_ID (probe/dissection ID), not Patient_ID
    % This is because the same patient may contribute multiple probes from
    % recidival tumors, and each probe should be treated independently.
    %
    % This implementation follows best practices from:
    % - Baker et al. (2014) Nature Protocols - data quality assurance
    % - Greener et al. (2022) Nature Rev. Mol. Cell Biol. - ML best practices
    
    fprintf('  Validating %s dataset...\n', dataset_name);
    
    if isempty(patientData)
        error('Patient data is empty!');
    end
    
    % Check 1: Unique probe IDs (Diss_ID)
    probeIDs = {patientData.patientID};  % Actually contains Diss_ID
    uniqueIDs = unique(probeIDs);
    if length(uniqueIDs) ~= length(probeIDs)
        % Find duplicates
        [~, idx] = unique(probeIDs, 'first');
        duplicate_idx = setdiff(1:length(probeIDs), idx);
        duplicate_ids = probeIDs(duplicate_idx);
        fprintf('WARNING: Found %d duplicate probe IDs:\n', length(duplicate_ids));
        for d = 1:min(10, length(duplicate_ids))
            fprintf('  - %s\n', duplicate_ids{d});
        end
        error('Probe IDs (Diss_ID) are not unique! Found %d duplicates out of %d entries.', ...
              length(probeIDs) - length(uniqueIDs), length(probeIDs));
    end
    fprintf('    ✓ All probe IDs are unique (%d probes/samples)\n', length(probeIDs));
    
    % Check 2: Valid labels (1 or 3 only)
    labels = [patientData.label];
    if ~all(ismember(labels, [1, 3]))
        invalid_labels = unique(labels(~ismember(labels, [1, 3])));
        error('Invalid labels found: %s. Only 1 (WHO-1) and 3 (WHO-3) allowed.', ...
              mat2str(invalid_labels));
    end
    fprintf('    ✓ All labels are valid (WHO-1 or WHO-3)\n');
    
    % Check 3: Class balance
    nWHO1 = sum(labels == 1);
    nWHO3 = sum(labels == 3);
    fprintf('    ✓ Class distribution: WHO-1=%d patients, WHO-3=%d patients\n', nWHO1, nWHO3);
    
    % Check 4: Spectra dimensions consistency
    nWavenumbers = size(patientData(1).spectra, 2);
    fprintf('    Checking spectra dimensions (expecting %d wavenumbers)...\n', nWavenumbers);
    
    for i = 1:length(patientData)
        % Check wavenumber dimension
        if size(patientData(i).spectra, 2) ~= nWavenumbers
            error('Patient %s has inconsistent wavenumber dimension: expected %d, got %d', ...
                  patientData(i).patientID, nWavenumbers, size(patientData(i).spectra, 2));
        end
        
        % Check for empty spectra
        if size(patientData(i).spectra, 1) == 0
            error('Patient %s has no spectra!', patientData(i).patientID);
        end
        
        % Check for NaN/Inf values
        if any(isnan(patientData(i).spectra(:)))
            error('Patient %s contains NaN values in spectra', patientData(i).patientID);
        end
        if any(isinf(patientData(i).spectra(:)))
            error('Patient %s contains Inf values in spectra', patientData(i).patientID);
        end
    end
    fprintf('    ✓ All spectra have consistent dimensions (%d wavenumbers)\n', nWavenumbers);
    fprintf('    ✓ No NaN/Inf values detected\n');
    
    % Check 5: Metadata completeness
    fprintf('    Checking metadata completeness...\n');
    for i = 1:length(patientData)
        if ~isfield(patientData(i), 'metadata') || isempty(patientData(i).metadata)
            warning('Patient %s has missing metadata', patientData(i).patientID);
        end
    end
    fprintf('    ✓ Metadata check complete\n');
    
    fprintf('  ✓ %s dataset validation PASSED\n', dataset_name);
end
