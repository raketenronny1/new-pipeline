%% CONFIG - Configuration settings for Meningioma FT-IR Classification Pipeline
%
% Central configuration file for all pipeline parameters including:
%   - Data paths
%   - Quality control thresholds
%   - PCA settings
%   - Cross-validation parameters
%   - Hyperparameter optimization settings
%   - Classifier configurations
%   - Cost-sensitive learning parameters
%
% SYNTAX:
%   cfg = config()
%
% OUTPUTS:
%   cfg - Structure containing all configuration parameters
%
% USAGE:
%   cfg = config();
%   data = load_data_direct(cfg);
%
% IMPORTANT PARAMETERS:
%   - cfg.classifiers.cost_who3_penalty: Controls cost-sensitive learning
%     Higher values (e.g., 5-10) prioritize WHO-3 detection
%     Lower values (e.g., 1-3) provide more balanced classification
%
% See also: load_data_direct, evaluate_test_set_direct, run_patientwise_cv_direct

function cfg = config()
    % Meningioma FT-IR Classification Pipeline Configuration
    
    cfg = struct();
    
    % === PATHS ===
    cfg.paths.data = 'data/';
    cfg.paths.models = 'models/meningioma_ftir_pipeline/';
    cfg.paths.results = 'results/meningioma_ftir_pipeline/';
    cfg.paths.qc = 'results/meningioma_ftir_pipeline/qc/';
    cfg.paths.eda = 'results/eda/';  % EDA results (PCA model, outlier flags)
    
    % === QUALITY CONTROL ===
    % Thresholds for spectrum and sample quality filtering
    cfg.qc = struct();
    cfg.qc.snr_threshold = 10;                      % Minimum signal-to-noise ratio
    cfg.qc.max_absorbance = 1.8;                    % Maximum allowed absorbance
    cfg.qc.baseline_sd_threshold = 0.02;            % Maximum baseline standard deviation
    cfg.qc.min_absorbance = -0.1;                   % Minimum allowed absorbance
    cfg.qc.amide_ratio_min = 1.2;                   % Minimum Amide I/II ratio
    cfg.qc.amide_ratio_max = 3.5;                   % Maximum Amide I/II ratio
    cfg.qc.within_sample_corr_threshold = 0.85;     % Min correlation within sample
    cfg.qc.min_spectra_per_sample = 100;            % Min spectra required per sample
    cfg.qc.outlier_confidence = 0.99;               % Chi-squared confidence for outliers
    cfg.qc.max_samples_to_exclude = 2;              % Conservative exclusion limit
    
    % === PCA (APPLIED ONLY FOR LDA) ===
    % Principal Component Analysis settings for dimensionality reduction
    cfg.pca.variance_threshold = 0.95;              % Retain 95% of variance
    cfg.pca.max_components = 15;                    % Maximum number of components
    
    % === CROSS-VALIDATION ===
    % Settings for model evaluation and validation
    cfg.cv.n_folds = 5;                             % K-fold cross-validation
    cfg.cv.n_repeats = 50;                          % Number of CV repetitions
    cfg.cv.stratified = true;                       % Maintain class balance in folds
    
    % === HYPERPARAMETER OPTIMIZATION ===
    % Bayesian optimization settings for classifier tuning
    cfg.optimization.enabled = true;                % Enable/disable optimization
    cfg.optimization.mode = 'all';                  % 'all', 'selective', or 'none'
    cfg.optimization.classifiers_to_optimize = {'LDA', 'PLSDA', 'SVM', 'RandomForest'};
    cfg.optimization.max_evaluations = 30;          % Bayesian optimization iterations
    cfg.optimization.use_parallel = true;           % Use parallel processing
    cfg.optimization.kfold_inner = 3;               % Inner CV folds (faster than 5)
    cfg.optimization.verbose = 1;                   % 0=quiet, 1=progress, 2=detailed
    
    % Optimization ranges for each classifier
    cfg.optimization.lda_delta_range = [0, 1];      % Regularization for linear covariance
    cfg.optimization.lda_gamma_range = [0, 1];      % Regularization for quadratic term
    cfg.optimization.plsda_components = 1:15;       % PLS components to test
    cfg.optimization.svm_box_range = [0.1, 100];    % BoxConstraint (C parameter)
    cfg.optimization.svm_kernel_range = [0.001, 10]; % KernelScale (gamma parameter)
    cfg.optimization.rf_trees = [50, 500];          % Number of trees
    cfg.optimization.rf_leaf_size = [1, 50];        % Minimum leaf size
    
    % === CLASSIFIERS ===
    % Configuration for all supported classification algorithms
    cfg.classifiers.types = {'LDA', 'PLSDA', 'SVM', 'RandomForest'};
    
    % Primary classifier for patient-wise CV
    cfg.classifiers.primary_type = 'SVM';
    
    % SVM hyperparameters (for patient-wise CV)
    cfg.classifiers.svm_C = 1;                      % Box constraint (regularization)
    cfg.classifiers.svm_gamma = 'auto';             % RBF kernel scale (1/n_features)
    
    % RandomForest hyperparameters (for patient-wise CV)
    cfg.classifiers.rf_n_trees = 100;               % Number of trees in forest
    cfg.classifiers.rf_min_leaf_size = 5;           % Minimum leaf size
    
    % Hyperparameter grids (for original pipeline with grid search)
    cfg.classifiers.plsda_n_components = 1:10;
    cfg.classifiers.svm_C_values = 10.^(-2:0.5:2);
    cfg.classifiers.svm_gamma_values = 10.^(-3:0.5:1);
    cfg.classifiers.rf_n_trees_grid = [50, 100, 200, 500];
    cfg.classifiers.rf_max_depth = [5, 10, 20, 30];
    
    % === COST-SENSITIVE LEARNING ===
    % Prioritize detection of malignant WHO-3 tumors
    cfg.classifiers.cost_sensitive = true;          % Enable cost-sensitive learning
    cfg.classifiers.cost_who3_penalty = 5;          % Penalty for missing WHO-3 (1-10)
    % Higher penalty values prioritize WHO-3 detection (fewer false negatives)
    % Lower values provide more balanced classification
    % Recommended range: 3-7 for clinical applications
    
    % === REPRODUCIBILITY ===
    cfg.random_seed = 42;
    
    % === REPORTING ===
    cfg.report.figures_format = 'png';
    cfg.report.figures_dpi = 300;
    cfg.report.save_intermediate = true;
end