function test_integration_mini()
    %TEST_INTEGRATION_MINI Mini integration test with synthetic data
    %
    % Tests complete pipeline with small synthetic dataset:
    %   - DataLoader
    %   - PreprocessingPipeline
    %   - ClassifierWrapper
    %   - CrossValidationEngine
    %   - MetricsCalculator
    %   - ResultsAggregator
    %   - ReportGenerator
    
    fprintf('\n========================================\n');
    fprintf('MINI INTEGRATION TEST\n');
    fprintf('========================================\n\n');
    
    % Setup paths
    addpath(fullfile(pwd, '../src/utils'));
    addpath(fullfile(pwd, '../src/preprocessing'));
    addpath(fullfile(pwd, '../src/classifiers'));
    addpath(fullfile(pwd, '../src/validation'));
    addpath(fullfile(pwd, '../src/metrics'));
    addpath(fullfile(pwd, '../src/reporting'));
    
    %% 1. Create synthetic dataset
    fprintf('1. Creating synthetic dataset...\n');
    
    n_patients = 12;  % Minimum for 3-fold CV
    samples_per_patient = 3;
    n_samples = n_patients * samples_per_patient;
    n_features = 20;
    
    % Create patient IDs
    patient_ids = repelem(1:n_patients, samples_per_patient)';
    
    % Create 2-class labels (stratified)
    y = categorical(mod(patient_ids, 2) + 1);
    
    % Create separable features
    X = zeros(n_samples, n_features);
    for i = 1:n_samples
        if y(i) == categorical(1)
            X(i, :) = randn(1, n_features) + 1;  % Class 1: positive offset
        else
            X(i, :) = randn(1, n_features) - 1;  % Class 2: negative offset
        end
    end
    
    fprintf('   Created: %d samples, %d patients, %d features\n', ...
        n_samples, n_patients, n_features);
    
    %% 2. Configure pipeline
    fprintf('2. Configuring pipeline...\n');
    
    cfg = struct();
    cfg.n_folds = 3;
    cfg.n_repeats = 2;
    cfg.random_seed = 42;
    cfg.parallel = false;
    
    % Minimal preprocessing (2 permutations)
    cfg.preprocessing_permutations = {'10200X', '10220X'};  % Norm, Norm+2nd deriv
    
    % 2 classifiers
    cfg.classifiers = {'PCA-LDA', 'SVM-RBF'};
    
    % PCA-LDA parameters
    cfg.pca_variance_threshold = 0.95;
    cfg.pca_max_components = 10;
    
    % SVM parameters
    cfg.svm_C = 1.0;
    cfg.svm_kernel_scale = 'auto';  % Changed from svm_gamma
    
    % PLS-DA parameters
    cfg.plsda_n_components = 5;
    
    % RandomForest parameters
    cfg.rf_n_trees = 50;
    cfg.rf_min_leaf_size = 1;
    
    fprintf('   Permutations: %d, Classifiers: %d, Folds: %d, Repeats: %d\n', ...
        length(cfg.preprocessing_permutations), length(cfg.classifiers), ...
        cfg.n_folds, cfg.n_repeats);
    
    %% 3. Run Cross-Validation
    fprintf('3. Running cross-validation...\n');
    
    cv_engine = CrossValidationEngine(cfg, 'Verbose', false);
    cv_results = cv_engine.run(X, y, patient_ids);
    
    fprintf('   CV complete: %d configurations tested\n', ...
        cv_results.n_permutations * cv_results.n_classifiers);
    
    %% 4. Aggregate Results
    fprintf('4. Aggregating results...\n');
    
    aggregator = ResultsAggregator(cv_results, 'Verbose', false);
    summary_spectrum = aggregator.summarize('Level', 'spectrum');
    summary_patient = aggregator.summarize('Level', 'patient');
    
    fprintf('   Spectrum-level summary: %d configs\n', numel(summary_spectrum.configurations));
    fprintf('   Patient-level summary: %d configs\n', numel(summary_patient.configurations));
    
    %% 5. Find Best Configuration
    fprintf('5. Identifying best configurations...\n');
    
    best_spectrum = aggregator.get_best_configuration('accuracy', 'Level', 'spectrum');
    best_patient = aggregator.get_best_configuration('accuracy', 'Level', 'patient');
    
    fprintf('   Best (spectrum): %s + %s = %.4f\n', ...
        best_spectrum.permutation_id, best_spectrum.classifier_name, best_spectrum.best_value);
    fprintf('   Best (patient): %s + %s = %.4f\n', ...
        best_patient.permutation_id, best_patient.classifier_name, best_patient.best_value);
    
    %% 6. Generate Report
    fprintf('6. Generating report...\n');
    
    output_dir = fullfile(pwd, 'integration_test_output');
    reporter = ReportGenerator(cv_results, 'OutputDir', output_dir, ...
        'Verbose', false, 'SavePlots', true);
    reporter.generate_full_report();
    
    fprintf('   Report saved to: %s\n', output_dir);
    
    %% 7. Validation Checks
    fprintf('7. Validating results...\n');
    
    % Check all configurations were processed
    assert(numel(summary_spectrum.configurations) == ...
        cv_results.n_permutations * cv_results.n_classifiers, ...
        'Not all configurations processed');
    
    % Check accuracy is reasonable (>0.4 for separable data)
    assert(best_spectrum.best_value > 0.4, ...
        sprintf('Accuracy too low: %.4f', best_spectrum.best_value));
    
    % Check output files exist
    assert(exist(fullfile(output_dir, 'spectrum_level_summary.mat'), 'file') > 0, ...
        'Spectrum summary not saved');
    assert(exist(fullfile(output_dir, 'patient_level_summary.mat'), 'file') > 0, ...
        'Patient summary not saved');
    assert(exist(fullfile(output_dir, 'analysis_summary.txt'), 'file') > 0, ...
        'Text report not saved');
    
    % Check CSV tables
    tbl_spectrum = readtable(fullfile(output_dir, 'spectrum_level_results.csv'));
    assert(height(tbl_spectrum) == cv_results.n_permutations * cv_results.n_classifiers, ...
        'Wrong number of rows in spectrum table');
    
    fprintf('   All validation checks PASSED\n');
    
    %% 8. Display Summary Statistics
    fprintf('\n8. Summary Statistics:\n');
    fprintf('   =====================================\n');
    
    for p = 1:size(summary_spectrum.configurations, 1)
        for c = 1:size(summary_spectrum.configurations, 2)
            config = summary_spectrum.configurations{p, c};
            fprintf('   %s + %s:\n', config.permutation_id, config.classifier_name);
            fprintf('     Accuracy: %.4f ± %.4f\n', ...
                config.mean_metrics.accuracy, config.std_metrics.accuracy);
            fprintf('     F1 Score: %.4f ± %.4f\n', ...
                config.mean_metrics.macro_f1, config.std_metrics.macro_f1);
            fprintf('     AUC:      %.4f ± %.4f\n', ...
                config.mean_metrics.auc, config.std_metrics.auc);
        end
    end
    
    fprintf('\n========================================\n');
    fprintf('INTEGRATION TEST: COMPLETE ✓\n');
    fprintf('========================================\n');
    
    % Cleanup (optional - comment out to inspect results)
    % rmdir(output_dir, 's');
end
