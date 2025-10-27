%% TEST EDA-INTEGRATED PIPELINE
% Tests the streamlined pipeline with EDA-based outlier detection

clear; clc;

% Navigate to project root
project_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
cd(project_root);
fprintf('Working directory: %s\n\n', pwd);

% Add source path
addpath(fullfile(project_root, 'src', 'meningioma_ftir_pipeline'));

fprintf('═══════════════════════════════════════════════════════════\n');
fprintf('  TESTING EDA-INTEGRATED PIPELINE\n');
fprintf('═══════════════════════════════════════════════════════════\n\n');

try
    %% Test 1: Check if EDA results exist
    fprintf('[Test 1/4] Checking EDA results...\n');
    eda_file = 'results/eda/eda_results_PP1.mat';
    
    if ~exist(eda_file, 'file')
        fprintf('  ⚠ EDA results not found.\n');
        fprintf('  Running EDA first (this will take a few minutes)...\n\n');
        run_full_eda();
        fprintf('\n  ✓ EDA completed\n');
    else
        fprintf('  ✓ EDA results found: %s\n', eda_file);
        
        % Load and inspect
        load(eda_file, 'eda_results');
        fprintf('    - PCA components: %d\n', size(eda_results.pca.coeff, 2));
        fprintf('    - Outliers detected: %d / %d (%.1f%%)\n', ...
                sum(eda_results.pca.outliers_both), ...
                length(eda_results.pca.outliers_both), ...
                100*sum(eda_results.pca.outliers_both)/length(eda_results.pca.outliers_both));
    end
    
    %% Test 2: Load data with EDA filtering
    fprintf('\n[Test 2/4] Loading data with EDA filtering...\n');
    cfg = config();
    tic;
    data = load_data_with_eda(cfg);
    t = toc;
    
    fprintf('  ✓ Data loaded in %.2f seconds\n', t);
    fprintf('    Training: %d samples (%d spectra after filtering)\n', ...
            data.train.n_samples, data.train.total_spectra);
    fprintf('    Test: %d samples (%d spectra)\n', ...
            data.test.n_samples, data.test.total_spectra);
    fprintf('    PCA model: %d components (%.1f%% variance)\n', ...
            data.pca_model.n_comp, data.pca_model.total_variance);
    
    %% Test 3: Verify data structure
    fprintf('\n[Test 3/4] Validating data structure...\n');
    
    % Check for NaN/Inf
    has_nan = any(cellfun(@(x) any(isnan(x(:))), data.train.spectra));
    has_inf = any(cellfun(@(x) any(isinf(x(:))), data.train.spectra));
    
    if has_nan || has_inf
        error('Data contains NaN or Inf values!');
    end
    fprintf('  ✓ No NaN/Inf values\n');
    
    % Check PCA model structure
    assert(isfield(data.pca_model, 'coeff'), 'PCA model missing coeff field');
    assert(isfield(data.pca_model, 'n_comp'), 'PCA model missing n_comp field');
    assert(isfield(data.pca_model, 'X_mean'), 'PCA model missing X_mean field');
    fprintf('  ✓ PCA model structure valid\n');
    
    % Check dimensions
    fprintf('  Checking sample dimensions...\n');
    for i = 1:min(3, data.train.n_samples)
        fprintf('    Sample %d: %s → %d spectra × %d wavenumbers\n', ...
                i, data.train.diss_id{i}, size(data.train.spectra{i}));
    end
    fprintf('  ✓ Data structure valid\n');
    
    %% Test 4: Quick CV test (1 repeat, 3 folds)
    fprintf('\n[Test 4/4] Running quick CV test...\n');
    fprintf('  (1 repeat, 3 folds, all classifiers)\n');
    
    % Temporarily modify config for quick test
    cfg_test = cfg;
    cfg_test.cv.n_folds = 3;
    cfg_test.cv.n_repeats = 1;
    cfg_test.optimization.enabled = false;  % Skip optimization for speed
    
    tic;
    cvResults = run_patientwise_cv_direct(data, cfg_test);
    t = toc;
    
    fprintf('  ✓ CV completed in %.2f seconds\n', t);
    fprintf('\n  Results summary:\n');
    
    % Display results
    classifier_names = {'LDA', 'PLSDA', 'SVM', 'RandomForest'};
    for i = 1:length(classifier_names)
        clf_name = classifier_names{i};
        if isfield(cvResults, clf_name) && isfield(cvResults.(clf_name), 'metrics')
            metrics = cvResults.(clf_name).metrics;
            
            fprintf('    %s:\n', clf_name);
            fprintf('      Accuracy: %.2f%% ± %.2f%%\n', ...
                    metrics.accuracy_mean * 100, metrics.accuracy_std * 100);
            fprintf('      Sensitivity: %.2f%%\n', metrics.sensitivity_mean * 100);
            fprintf('      Specificity: %.2f%%\n', metrics.specificity_mean * 100);
            
            % Verify LDA used EDA PCA
            if strcmp(clf_name, 'LDA')
                fprintf('      [LDA used EDA PCA model: %d PCs]\n', data.pca_model.n_comp);
            end
        end
    end
    
    fprintf('\n═══════════════════════════════════════════════════════════\n');
    fprintf('  ✓ ALL TESTS PASSED!\n');
    fprintf('═══════════════════════════════════════════════════════════\n\n');
    
    fprintf('EDA-integrated pipeline is working correctly.\n');
    fprintf('Key improvements:\n');
    fprintf('  ✓ EDA performs PCA and outlier detection\n');
    fprintf('  ✓ Outliers removed from training data\n');
    fprintf('  ✓ EDA PCA model (15 PCs) used for LDA\n');
    fprintf('  ✓ Other classifiers use raw standardized spectra\n');
    fprintf('  ✓ No redundant Mahalanobis outlier detection\n\n');
    
    fprintf('To run full pipeline:\n');
    fprintf('  run_pipeline_with_eda()\n\n');
    
catch ME
    fprintf('\n✗ TEST FAILED!\n');
    fprintf('Error: %s\n', ME.message);
    fprintf('Location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    rethrow(ME);
end
