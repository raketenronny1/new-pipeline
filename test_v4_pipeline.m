%% Test Script for v4.0 Unified Pipeline
% Quick validation test with reduced CV settings
%
% This script tests the refactored pipeline with:
% - NFolds = 3 (reduced from default 5)
% - NRepeats = 5 (reduced from default 50)
% - All classifiers enabled
% - EDA-based outlier detection

clear; close all; clc;

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║  Testing v4.0 Unified Pipeline                            ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n');
fprintf('\n');

%% Add paths
addpath('src/meningioma_ftir_pipeline');
addpath('src/utils');
addpath('src/preprocessing');

%% Test Configuration
fprintf('TEST CONFIGURATION:\n');
fprintf('  NFolds:       3\n');
fprintf('  NRepeats:     5\n');
fprintf('  Classifiers:  LDA, PLSDA, SVM, RandomForest\n');
fprintf('  EDA:          Enabled\n');
fprintf('  Outliers:     EDA-based (T²-Q statistics)\n');
fprintf('\n');

%% Run Pipeline
try
    fprintf('Starting pipeline test...\n');
    fprintf('═══════════════════════════════════════════════════════════\n\n');
    
    % Start timer
    test_start = tic;
    
    % Run with reduced settings for quick validation
    results = run_pipeline('NFolds', 3, ...
                          'NRepeats', 5, ...
                          'RunEDA', true, ...
                          'OutlierMethod', 'eda', ...
                          'Verbose', true, ...
                          'SaveResults', true);
    
    % Total test time
    test_time = toc(test_start);
    
    %% Validate Results
    fprintf('\n');
    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║  VALIDATION CHECKS                                        ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n\n');
    
    % Check 1: Results structure
    fprintf('✓ Check 1: Results structure returned\n');
    assert(isstruct(results), 'Results should be a structure');
    assert(isfield(results, 'cv_results'), 'Missing cv_results field');
    assert(isfield(results, 'data'), 'Missing data field');
    assert(isfield(results, 'config'), 'Missing config field');
    assert(isfield(results, 'timestamp'), 'Missing timestamp field');
    fprintf('  Fields: cv_results, data, config, timestamp ✓\n\n');
    
    % Check 2: All classifiers present
    fprintf('✓ Check 2: All classifiers executed\n');
    expected_classifiers = {'LDA', 'PLSDA', 'SVM', 'RandomForest'};
    for i = 1:length(expected_classifiers)
        clf_name = expected_classifiers{i};
        assert(isfield(results.cv_results, clf_name), ...
               sprintf('Missing classifier: %s', clf_name));
        fprintf('  %s: ✓\n', clf_name);
    end
    fprintf('\n');
    
    % Check 3: Metrics computed
    fprintf('✓ Check 3: Performance metrics computed\n');
    for i = 1:length(expected_classifiers)
        clf_name = expected_classifiers{i};
        m = results.cv_results.(clf_name).metrics;
        
        % Check all required metrics exist
        assert(isfield(m, 'accuracy_mean'), 'Missing accuracy_mean');
        assert(isfield(m, 'sensitivity_mean'), 'Missing sensitivity_mean');
        assert(isfield(m, 'specificity_mean'), 'Missing specificity_mean');
        assert(isfield(m, 'auc_mean'), 'Missing auc_mean');
        
        % Check values are in valid range [0, 1]
        assert(m.accuracy_mean >= 0 && m.accuracy_mean <= 1, ...
               'Accuracy out of range');
        assert(m.sensitivity_mean >= 0 && m.sensitivity_mean <= 1, ...
               'Sensitivity out of range');
        assert(m.specificity_mean >= 0 && m.specificity_mean <= 1, ...
               'Specificity out of range');
        assert(m.auc_mean >= 0 && m.auc_mean <= 1, ...
               'AUC out of range');
        
        fprintf('  %s: Acc=%.3f, Sen=%.3f, Spe=%.3f, AUC=%.3f ✓\n', ...
                clf_name, m.accuracy_mean, m.sensitivity_mean, ...
                m.specificity_mean, m.auc_mean);
    end
    fprintf('\n');
    
    % Check 4: Data structure
    fprintf('✓ Check 4: Data structure validated\n');
    assert(isfield(results.data, 'train'), 'Missing train data');
    assert(isfield(results.data, 'test'), 'Missing test data');
    assert(isfield(results.data.train, 'n_samples'), 'Missing n_samples');
    fprintf('  Training samples: %d\n', results.data.train.n_samples);
    fprintf('  Test samples:     %d\n', results.data.test.n_samples);
    fprintf('\n');
    
    % Check 5: Files created
    fprintf('✓ Check 5: Output files created\n');
    results_dir = results.config.paths.results;
    assert(exist(results_dir, 'dir') == 7, 'Results directory not created');
    
    % Check for expected output files
    expected_files = {'cv_results.mat', ...
                      'cross_validation_summary.txt', ...
                      'detailed_predictions_LDA.xlsx'};
    
    for i = 1:length(expected_files)
        file_path = fullfile(results_dir, expected_files{i});
        if exist(file_path, 'file')
            fprintf('  %s ✓\n', expected_files{i});
        else
            fprintf('  %s (not found)\n', expected_files{i});
        end
    end
    fprintf('\n');
    
    %% Summary
    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║  TEST SUMMARY                                             ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n\n');
    fprintf('✓ ALL VALIDATION CHECKS PASSED\n\n');
    fprintf('Test Duration: %.1f minutes\n', test_time / 60);
    fprintf('Results saved to: %s\n\n', results_dir);
    
    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║  v4.0 PIPELINE VALIDATED SUCCESSFULLY                     ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n\n');
    
catch ME
    fprintf('\n');
    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║  TEST FAILED                                              ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n\n');
    fprintf('Error: %s\n', ME.message);
    fprintf('Location: %s (line %d)\n\n', ME.stack(1).name, ME.stack(1).line);
    
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  %d. %s (line %d)\n', i, ME.stack(i).name, ME.stack(i).line);
    end
    fprintf('\n');
    
    rethrow(ME);
end
