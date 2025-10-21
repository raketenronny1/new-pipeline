%% VALIDATION AND TEST SCRIPT
% Quick validation script to test the patient-wise CV implementation
%
% This script performs basic checks and runs a simplified version
% of the pipeline to ensure everything is working correctly.

function test_patientwise_implementation()
    fprintf('\n');
    fprintf('═══════════════════════════════════════════════════════════\n');
    fprintf('  PATIENT-WISE CV IMPLEMENTATION VALIDATION\n');
    fprintf('═══════════════════════════════════════════════════════════\n\n');
    
    %% Setup
    addpath('src/meningioma_ftir_pipeline');
    cfg = config();
    
    %% Test 1: Configuration Check
    fprintf('[Test 1/6] Checking configuration...\n');
    assert(isstruct(cfg), 'Config must be a struct');
    assert(isfield(cfg, 'paths'), 'Config must have paths field');
    assert(isfield(cfg, 'cv'), 'Config must have cv field');
    fprintf('  ✓ Configuration valid\n\n');
    
    %% Test 2: Data File Existence
    fprintf('[Test 2/6] Checking data files...\n');
    train_file = fullfile(cfg.paths.data, 'data_table_train.mat');
    test_file = fullfile(cfg.paths.data, 'data_table_test.mat');
    wn_file = fullfile(cfg.paths.data, 'wavenumbers.mat');
    
    if exist(train_file, 'file')
        fprintf('  ✓ Training data found\n');
    else
        error('Training data not found: %s', train_file);
    end
    
    if exist(test_file, 'file')
        fprintf('  ✓ Test data found\n');
    else
        error('Test data not found: %s', test_file);
    end
    
    if exist(wn_file, 'file')
        fprintf('  ✓ Wavenumber data found\n');
    else
        error('Wavenumber data not found: %s', wn_file);
    end
    fprintf('\n');
    
    %% Test 3: Load and Validate Raw Data Structure
    fprintf('[Test 3/6] Validating raw data structure...\n');
    load(train_file, 'dataTableTrain');
    
    fprintf('  Number of patients in training set: %d\n', height(dataTableTrain));
    
    % Check first patient
    first_patient_spectra = dataTableTrain.CombinedSpectra{1};
    fprintf('  First patient spectra size: %d × %d\n', size(first_patient_spectra, 1), size(first_patient_spectra, 2));
    
    % Check for WHO_Grade
    assert(ismember('WHO_Grade', dataTableTrain.Properties.VariableNames), ...
           'dataTableTrain must have WHO_Grade column');
    fprintf('  ✓ Raw data structure valid\n\n');
    
    %% Test 4: Patient-Wise Data Loading
    fprintf('[Test 4/6] Testing patient-wise data loading...\n');
    
    % Check if patient-wise data already exists
    patientwise_file = fullfile(cfg.paths.results, 'patientwise_data.mat');
    if exist(patientwise_file, 'file')
        fprintf('  Loading existing patient-wise data...\n');
        load(patientwise_file, 'trainingData', 'testData');
    else
        fprintf('  Creating patient-wise data structure...\n');
        load_and_prepare_data_patientwise(cfg);
        load(patientwise_file, 'trainingData', 'testData');
    end
    
    % Validate structure
    assert(isfield(trainingData, 'patientData'), 'trainingData must have patientData field');
    assert(length(trainingData.patientData) > 0, 'Must have at least one patient');
    
    fprintf('  Number of training patients: %d\n', length(trainingData.patientData));
    fprintf('  Number of test patients: %d\n', length(testData.patientData));
    
    % Check first patient
    first_pat = trainingData.patientData(1);
    fprintf('  First patient:\n');
    fprintf('    ID: %s\n', first_pat.patientID);
    fprintf('    Label: %d\n', first_pat.label);
    fprintf('    Spectra: %d × %d\n', size(first_pat.spectra, 1), size(first_pat.spectra, 2));
    
    assert(ismember(first_pat.label, [1, 3]), 'Label must be 1 or 3');
    assert(size(first_pat.spectra, 1) > 0, 'Patient must have at least one spectrum');
    fprintf('  ✓ Patient-wise data structure valid\n\n');
    
    %% Test 5: CV Fold Creation
    fprintf('[Test 5/6] Testing CV fold creation...\n');
    
    K = 3;  % Use 3 folds for quick testing
    fprintf('  Creating %d-fold patient-wise CV splits...\n', K);
    
    cvFolds = createPatientWiseStratifiedCV(trainingData.patientData, K, 42);
    
    % Validate folds
    assert(length(cvFolds) == K, sprintf('Must have %d folds', K));
    
    for k = 1:K
        trainIdx = cvFolds(k).trainPatientIdx;
        testIdx = cvFolds(k).testPatientIdx;
        
        % Check no overlap
        assert(isempty(intersect(trainIdx, testIdx)), ...
               sprintf('Fold %d has overlapping train/test patients!', k));
        
        % Check all patients accounted for
        allIdx = sort([trainIdx, testIdx]);
        assert(isequal(allIdx, 1:length(trainingData.patientData)), ...
               sprintf('Fold %d does not account for all patients', k));
    end
    
    fprintf('  ✓ CV folds created successfully\n');
    fprintf('  ✓ No patient overlap between train/test in any fold\n\n');
    
    %% Test 6: Extract Spectra for One Fold
    fprintf('[Test 6/6] Testing spectrum extraction...\n');
    
    [X_train, y_train, X_test, y_test, testPatientIDs, ~] = ...
        extractSpectraForFold(trainingData.patientData, ...
                              cvFolds(1).trainPatientIdx, ...
                              cvFolds(1).testPatientIdx);
    
    fprintf('  Fold 1 extraction:\n');
    fprintf('    Training spectra: %d\n', size(X_train, 1));
    fprintf('    Test spectra: %d\n', size(X_test, 1));
    fprintf('    Test patients: %d unique IDs\n', length(unique(testPatientIDs)));
    
    % Validate
    assert(size(X_train, 1) == length(y_train), 'X_train and y_train size mismatch');
    assert(size(X_test, 1) == length(y_test), 'X_test and y_test size mismatch');
    assert(size(X_test, 1) == length(testPatientIDs), 'Test size mismatch');
    assert(all(ismember(y_train, [1, 3])), 'Invalid training labels');
    assert(all(ismember(y_test, [1, 3])), 'Invalid test labels');
    
    fprintf('  ✓ Spectrum extraction successful\n');
    fprintf('  ✓ All labels valid\n\n');
    
    %% Summary
    fprintf('═══════════════════════════════════════════════════════════\n');
    fprintf('  ALL VALIDATION TESTS PASSED ✓\n');
    fprintf('═══════════════════════════════════════════════════════════\n\n');
    
    fprintf('The patient-wise CV implementation is ready to use!\n\n');
    
    fprintf('Next steps:\n');
    fprintf('  1. Run quality control (optional):\n');
    fprintf('       quality_control_analysis(config())\n\n');
    fprintf('  2. Run full patient-wise pipeline:\n');
    fprintf('       run_full_pipeline_patientwise()\n\n');
    fprintf('  3. Or run CV only:\n');
    fprintf('       cvResults = run_patientwise_cross_validation(config())\n\n');
    
    fprintf('═══════════════════════════════════════════════════════════\n\n');
end
