function test_cross_validation_engine()
    %TEST_CROSS_VALIDATION_ENGINE Unit tests for CrossValidationEngine
    %
    % TESTS:
    %   1. Constructor
    %   2. Patient-level CV partitioning
    %   3. No patient overlap validation
    %   4. Full CV run (serial mode, minimal)
    %   5. Results structure validity
    %   6. Reproducibility with random seed
    %   7. Stratification verification
    %   8. Multi-repeat consistency
    
    fprintf('\n=== TESTING CROSS-VALIDATION ENGINE ===\n\n');
    
    % Setup
    addpath(fullfile(pwd, '../src/utils'));
    addpath(fullfile(pwd, '../src/preprocessing'));
    addpath(fullfile(pwd, '../src/classifiers'));
    addpath(fullfile(pwd, '../src/validation'));
    
    % Test 1: Constructor
    fprintf('Test 1: Constructor... ');
    try
        cfg = createTestConfig();
        cv_engine = CrossValidationEngine(cfg, 'Verbose', false);
        assert(isa(cv_engine, 'CrossValidationEngine'), 'Object not created');
        fprintf('PASS\n');
    catch ME
        fprintf('FAIL: %s\n', ME.message);
    end
    
    % Test 2: Patient-level CV partitioning
    fprintf('Test 2: Patient-level CV partitioning... ');
    try
        cfg = createTestConfig();
        cv_engine = CrossValidationEngine(cfg, 'Verbose', false);
        
        % Create test data with known patients
        n_patients = 12;
        samples_per_patient = 5;
        n_samples = n_patients * samples_per_patient;
        
        patient_ids = repelem(1:n_patients, samples_per_patient)';
        y = categorical(repmat([1; 1; 1; 2; 2; 2; 3; 3; 3; 1; 2; 3], samples_per_patient, 1));
        X = randn(n_samples, 10);
        
        % Run CV
        results = cv_engine.run(X, y, patient_ids);
        
        assert(results.n_patients == n_patients, 'Incorrect patient count');
        assert(results.n_samples == n_samples, 'Incorrect sample count');
        fprintf('PASS\n');
    catch ME
        fprintf('FAIL: %s\n', ME.message);
    end
    
    % Test 3: No patient overlap validation
    fprintf('Test 3: No patient overlap validation... ');
    try
        cfg = createTestConfig();
        cv_engine = CrossValidationEngine(cfg, 'Verbose', false);
        
        % Create test data
        n_patients = 15;
        samples_per_patient = 4;
        n_samples = n_patients * samples_per_patient;
        
        patient_ids = repelem(1:n_patients, samples_per_patient)';
        y = categorical(mod(patient_ids, 3) + 1);  % 3 classes
        X = randn(n_samples, 10);
        
        % Run CV
        results = cv_engine.run(X, y, patient_ids);
        
        % Verify no overlap in each fold
        for p = 1:length(results.permutations)
            perm = results.permutations{p};
            for c = 1:length(perm.classifiers)
                clf = perm.classifiers{c};
                for r = 1:length(clf.repeats)
                    repeat = clf.repeats{r};
                    for f = 1:length(repeat.folds)
                        fold = repeat.folds{f};
                        % Check patient counts
                        assert(fold.n_train_patients + fold.n_test_patients == n_patients, ...
                            'Patient count mismatch');
                    end
                end
            end
        end
        
        fprintf('PASS\n');
    catch ME
        fprintf('FAIL: %s\n', ME.message);
    end
    
    % Test 4: Full CV run (minimal)
    fprintf('Test 4: Full CV run (serial mode, minimal)... ');
    try
        cfg = createTestConfig();
        cv_engine = CrossValidationEngine(cfg, 'Verbose', false);
        
        % Create simple test data
        n_patients = 12;
        samples_per_patient = 3;
        n_samples = n_patients * samples_per_patient;
        
        patient_ids = repelem(1:n_patients, samples_per_patient)';
        y = categorical(mod(patient_ids, 2) + 1);  % Binary classification
        
        % Create separable data
        X = zeros(n_samples, 5);
        for i = 1:n_samples
            if y(i) == categorical(1)
                X(i, :) = randn(1, 5) + 2;  % Class 1: positive offset
            else
                X(i, :) = randn(1, 5) - 2;  % Class 2: negative offset
            end
        end
        
        % Run CV
        results = cv_engine.run(X, y, patient_ids);
        
        % Validate results structure
        assert(isfield(results, 'permutations'), 'Missing permutations field');
        assert(length(results.permutations) == 2, 'Wrong number of permutations');
        
        % Check first fold of first classifier
        fold1 = results.permutations{1}.classifiers{1}.repeats{1}.folds{1};
        assert(isfield(fold1, 'y_true'), 'Missing y_true');
        assert(isfield(fold1, 'y_pred'), 'Missing y_pred');
        assert(isfield(fold1, 'scores'), 'Missing scores');
        assert(length(fold1.y_true) == fold1.n_test_samples, 'Prediction length mismatch');
        
        fprintf('PASS\n');
    catch ME
        fprintf('FAIL: %s\n', ME.message);
    end
    
    % Test 5: Results structure validity
    fprintf('Test 5: Results structure validity... ');
    try
        cfg = createTestConfig();
        cv_engine = CrossValidationEngine(cfg, 'Verbose', false);
        
        % Create test data
        n_patients = 9;
        samples_per_patient = 3;
        n_samples = n_patients * samples_per_patient;
        
        patient_ids = repelem(1:n_patients, samples_per_patient)';
        y = categorical(mod(patient_ids, 3) + 1);
        X = randn(n_samples, 8);
        
        results = cv_engine.run(X, y, patient_ids);
        
        % Check top-level structure
        required_fields = {'config', 'n_samples', 'n_features', 'n_patients', ...
            'n_permutations', 'n_classifiers', 'n_folds', 'n_repeats', 'permutations'};
        for i = 1:length(required_fields)
            assert(isfield(results, required_fields{i}), ...
                sprintf('Missing field: %s', required_fields{i}));
        end
        
        % Check nested structure
        perm1 = results.permutations{1};
        assert(isfield(perm1, 'permutation_id'), 'Missing permutation_id');
        assert(isfield(perm1, 'classifiers'), 'Missing classifiers');
        
        clf1 = perm1.classifiers{1};
        assert(isfield(clf1, 'classifier_name'), 'Missing classifier_name');
        assert(isfield(clf1, 'repeats'), 'Missing repeats');
        
        repeat1 = clf1.repeats{1};
        assert(isfield(repeat1, 'folds'), 'Missing folds');
        
        fold1 = repeat1.folds{1};
        fold_fields = {'repeat', 'fold', 'n_train_samples', 'n_test_samples', ...
            'n_train_patients', 'n_test_patients', 'y_true', 'y_pred', 'scores', 'patient_ids'};
        for i = 1:length(fold_fields)
            assert(isfield(fold1, fold_fields{i}), ...
                sprintf('Missing fold field: %s', fold_fields{i}));
        end
        
        fprintf('PASS\n');
    catch ME
        fprintf('FAIL: %s\n', ME.message);
    end
    
    % Test 6: Reproducibility with random seed
    fprintf('Test 6: Reproducibility with random seed... ');
    try
        cfg1 = createTestConfig();
        cfg2 = createTestConfig();
        
        % Create test data
        n_patients = 12;
        samples_per_patient = 3;
        n_samples = n_patients * samples_per_patient;
        
        patient_ids = repelem(1:n_patients, samples_per_patient)';
        y = categorical(mod(patient_ids, 2) + 1);
        X = randn(n_samples, 5);
        
        % Run CV twice with same seed
        cv_engine1 = CrossValidationEngine(cfg1, 'Verbose', false);
        results1 = cv_engine1.run(X, y, patient_ids);
        
        cv_engine2 = CrossValidationEngine(cfg2, 'Verbose', false);
        results2 = cv_engine2.run(X, y, patient_ids);
        
        % Compare predictions from first fold
        y_pred1 = results1.permutations{1}.classifiers{1}.repeats{1}.folds{1}.y_pred;
        y_pred2 = results2.permutations{1}.classifiers{1}.repeats{1}.folds{1}.y_pred;
        
        assert(isequal(y_pred1, y_pred2), 'Results not reproducible');
        
        fprintf('PASS\n');
    catch ME
        fprintf('FAIL: %s\n', ME.message);
    end
    
    % Test 7: Stratification verification
    fprintf('Test 7: Stratification verification... ');
    try
        cfg = createTestConfig();
        cv_engine = CrossValidationEngine(cfg, 'Verbose', false);
        
        % Create imbalanced data (8:4 ratio)
        n_patients_c1 = 8;
        n_patients_c2 = 4;
        samples_per_patient = 3;
        
        patient_ids = [repelem(1:n_patients_c1, samples_per_patient)'; ...
                      repelem((n_patients_c1+1):(n_patients_c1+n_patients_c2), samples_per_patient)'];
        y = categorical([ones(n_patients_c1 * samples_per_patient, 1); ...
                        2*ones(n_patients_c2 * samples_per_patient, 1)]);
        X = randn(length(patient_ids), 5);
        
        results = cv_engine.run(X, y, patient_ids);
        
        % Check class distribution in folds is approximately balanced
        % With stratification, each fold should have roughly 2:1 ratio
        for p = 1:length(results.permutations)
            perm = results.permutations{p};
            for c = 1:length(perm.classifiers)
                clf = perm.classifiers{c};
                for r = 1:length(clf.repeats)
                    repeat = clf.repeats{r};
                    for f = 1:length(repeat.folds)
                        fold = repeat.folds{f};
                        y_test = fold.y_true;
                        n_c1 = sum(y_test == categorical(1));
                        n_c2 = sum(y_test == categorical(2));
                        % Both classes should be present
                        assert(n_c1 > 0 && n_c2 > 0, 'Class missing in fold');
                    end
                end
            end
        end
        
        fprintf('PASS\n');
    catch ME
        fprintf('FAIL: %s\n', ME.message);
    end
    
    % Test 8: Multi-repeat consistency
    fprintf('Test 8: Multi-repeat consistency... ');
    try
        cfg = createTestConfig();
        cv_engine = CrossValidationEngine(cfg, 'Verbose', false);
        
        % Create test data
        n_patients = 12;
        samples_per_patient = 3;
        n_samples = n_patients * samples_per_patient;
        
        patient_ids = repelem(1:n_patients, samples_per_patient)';
        y = categorical(mod(patient_ids, 2) + 1);
        X = randn(n_samples, 5);
        
        results = cv_engine.run(X, y, patient_ids);
        
        % Verify we have 2 repeats
        clf1 = results.permutations{1}.classifiers{1};
        assert(length(clf1.repeats) == 2, 'Wrong number of repeats');
        
        % Repeats should have different CV partitions
        y_pred_r1 = clf1.repeats{1}.folds{1}.y_pred;
        y_pred_r2 = clf1.repeats{2}.folds{1}.y_pred;
        
        % They might be different due to different partitioning
        % Just verify both exist and have correct length
        assert(~isempty(y_pred_r1), 'Repeat 1 predictions empty');
        assert(~isempty(y_pred_r2), 'Repeat 2 predictions empty');
        
        fprintf('PASS\n');
    catch ME
        fprintf('FAIL: %s\n', ME.message);
    end
    
    fprintf('\n=== ALL TESTS COMPLETED ===\n');
end

function cfg = createTestConfig()
    %CREATETESTCONFIG Create minimal test configuration
    
    cfg = struct();
    cfg.n_folds = 3;
    cfg.n_repeats = 2;
    cfg.random_seed = 42;
    cfg.parallel = false;  % Serial mode for testing
    
    % Minimal preprocessing permutations (2 permutations)
    cfg.preprocessing_permutations = {'10200X', '10220X'};  % Norm only, Norm+2nd deriv
    
    % Minimal classifiers (just PCA-LDA)
    cfg.classifiers = {'PCA-LDA'};  % Correct case-sensitive name
    
    % PCA-LDA parameters (matching ClassifierWrapper expectations)
    cfg.pca_variance_threshold = 0.95;
    cfg.pca_max_components = 10;
    
    % SVM parameters (defaults even though not used)
    cfg.svm_C = 1.0;
    cfg.svm_kernel_scale = 'auto';  % Changed from svm_gamma
    
    % PLS-DA parameters
    cfg.plsda_n_components = 5;
    
    % RandomForest parameters
    cfg.rf_n_trees = 50;
    cfg.rf_min_leaf_size = 1;
end
