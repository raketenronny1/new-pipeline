%% WHO MENINGIOMA FTIR CLASSIFICATION - QUICK START GUIDE
% Complete production-ready pipeline for patient-level cross-validation

%% ========================================================================
%% PART 0: VERIFY DATA PREPARATION
%% ========================================================================

fprintf('=== VERIFYING DATA FILES ===\n');

% Check files exist
assert(exist('data/data_table_train.mat', 'file') == 2, ...
    'data_table_train.mat not found!');
assert(exist('data/data_table_test.mat', 'file') == 2, ...
    'data_table_test.mat not found!');

% Load and check structure
train_check = load('data/data_table_train.mat');
test_check = load('data/data_table_test.mat');

train_vars = train_check.data_table_train.Properties.VariableNames;
test_vars = test_check.data_table_test.Properties.VariableNames;

% Verify raw spectra fields exist
has_raw_train = ismember('RawSpectra', train_vars) || ismember('MeanRawSpectrum', train_vars);
has_raw_test = ismember('RawSpectra', test_vars) || ismember('MeanRawSpectrum', test_vars);

assert(has_raw_train && has_raw_test, ...
    'Raw spectra fields not found! Run prepare_data.m first.');

% Verify preprocessed columns are removed
bad_cols = {'CombinedSpectra_PP1', 'CombinedSpectra_PP2', ...
            'MeanSpectrum_PP1', 'MeanSpectrum_PP2', 'CombinedRawSpectra'};
found_bad = intersect(bad_cols, train_vars);

if ~isempty(found_bad)
    error(['Old preprocessed columns still present: %s\n' ...
           'Run prepare_data.m to clean data files.'], strjoin(found_bad, ', '));
end

fprintf('✓ Data files verified (raw spectra only)\n\n');
clear train_check test_check train_vars test_vars bad_cols found_bad has_raw_train has_raw_test;

%% ========================================================================
%% PART 1: SETUP AND CONFIGURATION
%% ========================================================================

% Add paths
addpath('src/utils');
addpath('src/preprocessing');
addpath('src/classifiers');
addpath('src/validation');
addpath('src/metrics');
addpath('src/reporting');

% Create configuration
cfg = struct();

% Cross-validation settings
cfg.n_folds = 5;                    % 5-fold CV
cfg.n_repeats = 10;                 % 10 repeats for stability
cfg.random_seed = 42;               % Reproducibility
cfg.parallel = true;                % Use parallel execution

% Preprocessing permutations (BSNCX notation)
cfg.preprocessing_permutations = {
    '10200X',    % Normalization only
    '10210X',    % Normalization + 1st derivative
    '10220X',    % Normalization + 2nd derivative
    '11220X'     % Binning + Smoothing + Normalization + 2nd derivative
};

% Classifiers to evaluate
cfg.classifiers = {'PCA-LDA', 'SVM-RBF', 'PLS-DA', 'RandomForest'};

% PCA-LDA parameters
cfg.pca_variance_threshold = 0.95;  % Keep 95% variance
cfg.pca_max_components = 20;        % Max components

% SVM parameters
cfg.svm_C = 1.0;                    % Regularization
cfg.svm_kernel_scale = 'auto';      % Auto kernel scale

% PLS-DA parameters
cfg.plsda_n_components = 10;        % Number of components

% RandomForest parameters
cfg.rf_n_trees = 100;               % Number of trees
cfg.rf_min_leaf_size = 5;           % Minimum leaf size

%% ========================================================================
%% PART 2: LOAD DATA
%% ========================================================================

fprintf('\n=== LOADING DATA ===\n');

% Create data loader
loader = DataLoader();

% Load training data
[X_train, y_train, patient_ids_train] = loader.load(...
    'data/data_table_train.mat', ...
    'AggregationMethod', 'mean');  % Average spectra per sample

fprintf('Training: %d samples, %d patients, %d features\n', ...
    size(X_train, 1), length(unique(patient_ids_train)), size(X_train, 2));

% Load test data (for final evaluation)
[X_test, y_test, patient_ids_test] = loader.load(...
    'data/data_table_test.mat', ...
    'AggregationMethod', 'mean');

fprintf('Test: %d samples, %d patients, %d features\n', ...
    size(X_test, 1), length(unique(patient_ids_test)), size(X_test, 2));

% Validate no patient overlap
overlap = intersect(unique(patient_ids_train), unique(patient_ids_test));
assert(isempty(overlap), 'ERROR: Patient overlap detected!');
fprintf('✓ No patient overlap between train and test sets\n');

%% ========================================================================
%% PART 3: RUN CROSS-VALIDATION ON TRAINING SET
%% ========================================================================

fprintf('\n=== RUNNING CROSS-VALIDATION ===\n');
fprintf('Configurations: %d permutations × %d classifiers = %d total\n', ...
    length(cfg.preprocessing_permutations), length(cfg.classifiers), ...
    length(cfg.preprocessing_permutations) * length(cfg.classifiers));

% Create CV engine
cv_engine = CrossValidationEngine(cfg, 'Verbose', true);

% Run cross-validation
cv_results = cv_engine.run(X_train, y_train, patient_ids_train);

fprintf('\n✓ Cross-validation complete\n');

%% ========================================================================
%% PART 4: ANALYZE RESULTS
%% ========================================================================

fprintf('\n=== ANALYZING RESULTS ===\n');

% Create aggregator
aggregator = ResultsAggregator(cv_results, 'Verbose', true);

% Get spectrum-level summary
summary_spectrum = aggregator.summarize('Level', 'spectrum');

% Get patient-level summary
summary_patient = aggregator.summarize('Level', 'patient');

% Find best configuration (patient-level accuracy)
best_config = aggregator.get_best_configuration('accuracy', 'Level', 'patient');

fprintf('\n=== BEST CONFIGURATION ===\n');
fprintf('Permutation: %s\n', best_config.permutation_id);
fprintf('Classifier: %s\n', best_config.classifier_name);
fprintf('Patient-level Accuracy: %.4f ± %.4f\n', ...
    best_config.best_value, best_config.std_metrics.accuracy);
fprintf('Patient-level F1: %.4f ± %.4f\n', ...
    best_config.mean_metrics.macro_f1, best_config.std_metrics.macro_f1);
fprintf('Patient-level AUC: %.4f ± %.4f\n', ...
    best_config.mean_metrics.auc, best_config.std_metrics.auc);

%% ========================================================================
%% PART 5: GENERATE COMPREHENSIVE REPORT
%% ========================================================================

fprintf('\n=== GENERATING REPORT ===\n');

% Create output directory with timestamp
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
output_dir = fullfile('results', sprintf('report_%s', timestamp));

% Generate report
reporter = ReportGenerator(cv_results, ...
    'OutputDir', output_dir, ...
    'Verbose', true, ...
    'SavePlots', true);

reporter.generate_full_report();

fprintf('\n✓ Report saved to: %s\n', output_dir);

%% ========================================================================
%% PART 6: TRAIN FINAL MODEL ON FULL TRAINING SET
%% ========================================================================

fprintf('\n=== TRAINING FINAL MODEL ===\n');

% Use best configuration
best_permutation = best_config.permutation_id;
best_classifier = best_config.classifier_name;

fprintf('Using: %s + %s\n', best_permutation, best_classifier);

% Create preprocessing pipeline with best permutation
pipeline = PreprocessingPipeline(best_permutation, 'Verbose', true);

% Fit and transform training data
[X_train_processed, preproc_params] = pipeline.fit_transform(X_train);

% Train classifier on full training set
final_classifier = ClassifierWrapper(best_classifier, cfg, 'Verbose', true);
final_classifier.train(X_train_processed, y_train);

fprintf('✓ Final model trained on %d samples\n', size(X_train, 1));

%% ========================================================================
%% PART 7: EVALUATE ON INDEPENDENT TEST SET
%% ========================================================================

fprintf('\n=== EVALUATING ON TEST SET ===\n');

% Apply preprocessing to test data (using frozen parameters from training)
X_test_processed = pipeline.transform(X_test, preproc_params);

% Predict on test set
[y_test_pred, test_scores] = final_classifier.predict(X_test_processed);

% Calculate spectrum-level metrics
calc = MetricsCalculator('Verbose', false);
test_metrics_spectrum = calc.compute_spectrum_metrics(...
    y_test, y_test_pred, test_scores);

% Calculate patient-level metrics
test_metrics_patient = calc.compute_patient_metrics(...
    y_test, y_test_pred, test_scores, patient_ids_test);

% Display results
fprintf('\n=== TEST SET PERFORMANCE ===\n');
fprintf('\nSpectrum-level:\n');
fprintf('  Accuracy: %.4f\n', test_metrics_spectrum.accuracy);
fprintf('  F1 Score: %.4f\n', test_metrics_spectrum.macro_f1);
fprintf('  AUC:      %.4f\n', test_metrics_spectrum.auc);

fprintf('\nPatient-level:\n');
fprintf('  Accuracy: %.4f\n', test_metrics_patient.accuracy);
fprintf('  F1 Score: %.4f\n', test_metrics_patient.macro_f1);
fprintf('  AUC:      %.4f\n', test_metrics_patient.auc);

% Display confusion matrix
fprintf('\nPatient-level Confusion Matrix:\n');
disp(test_metrics_patient.confusion_matrix);

%% ========================================================================
%% PART 8: SAVE FINAL RESULTS
%% ========================================================================

fprintf('\n=== SAVING FINAL RESULTS ===\n');

% Save final model
model_dir = fullfile('models', sprintf('final_model_%s', timestamp));
if ~exist(model_dir, 'dir')
    mkdir(model_dir);
end

save(fullfile(model_dir, 'final_classifier.mat'), 'final_classifier', '-v7.3');
save(fullfile(model_dir, 'preprocessing_params.mat'), 'preproc_params', '-v7.3');
save(fullfile(model_dir, 'test_metrics.mat'), ...
    'test_metrics_spectrum', 'test_metrics_patient', '-v7.3');
save(fullfile(model_dir, 'best_config.mat'), 'best_config', '-v7.3');

% Save predictions
predictions = table(patient_ids_test, y_test, y_test_pred, test_scores, ...
    'VariableNames', {'Patient_ID', 'True_Label', 'Predicted_Label', 'Scores'});
writetable(predictions, fullfile(model_dir, 'test_predictions.csv'));

fprintf('✓ Final model saved to: %s\n', model_dir);

%% ========================================================================
%% PART 9: CREATE TEST SET VISUALIZATIONS
%% ========================================================================

fprintf('\n=== CREATING VISUALIZATIONS ===\n');

viz = VisualizationTools('OutputDir', fullfile(model_dir, 'plots'), ...
    'SavePlots', true, 'FigureFormat', 'png');

% Confusion matrix
class_names = cellstr(string(categories(y_test)));
viz.plot_confusion_matrix(test_metrics_patient.confusion_matrix, ...
    class_names, 'Title', 'Test Set Confusion Matrix (Patient-level)', ...
    'Normalize', false);

% ROC curve
viz.plot_roc_curve(y_test, test_scores, ...
    'Title', 'Test Set ROC Curve');

fprintf('✓ Visualizations saved to: %s\n', fullfile(model_dir, 'plots'));

%% ========================================================================
%% SUMMARY
%% ========================================================================

fprintf('\n========================================\n');
fprintf('PIPELINE EXECUTION COMPLETE\n');
fprintf('========================================\n');
fprintf('\nCross-Validation Results:\n');
fprintf('  Best Configuration: %s + %s\n', best_permutation, best_classifier);
fprintf('  CV Accuracy (Patient): %.4f ± %.4f\n', ...
    best_config.best_value, best_config.std_metrics.accuracy);
fprintf('\nTest Set Results:\n');
fprintf('  Test Accuracy (Patient): %.4f\n', test_metrics_patient.accuracy);
fprintf('  Test F1 (Patient): %.4f\n', test_metrics_patient.macro_f1);
fprintf('\nOutput Locations:\n');
fprintf('  CV Report: %s\n', output_dir);
fprintf('  Final Model: %s\n', model_dir);
fprintf('========================================\n');
