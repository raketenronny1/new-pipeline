function cfg = config()
    % Configuration for Meningioma FT-IR Classification Pipeline
    
    cfg = struct();
    
    % === PATHS ===
    cfg.paths.data = 'data/';
    cfg.paths.models = 'models/meningioma_ftir_pipeline/';
    cfg.paths.results = 'results/meningioma_ftir_pipeline/';
    cfg.paths.qc = 'results/meningioma_ftir_pipeline/qc/';
    
    % === QUALITY CONTROL ===
    cfg.qc = struct();
    cfg.qc.snr_threshold = 10;
    cfg.qc.max_absorbance = 1.8;
    cfg.qc.baseline_sd_threshold = 0.02;
    cfg.qc.min_absorbance = -0.1;
    cfg.qc.amide_ratio_min = 1.2;
    cfg.qc.amide_ratio_max = 3.5;
    cfg.qc.within_sample_corr_threshold = 0.85;
    cfg.qc.min_spectra_per_sample = 100;
    cfg.qc.outlier_confidence = 0.99;  % Chi-squared threshold
    cfg.qc.max_samples_to_exclude = 2;  % Conservative limit
    
    % === PCA ===
    cfg.pca.variance_threshold = 0.95;  % Keep PCs explaining 95% variance
    cfg.pca.max_components = 15;  % Upper limit on components
    
    % === CROSS-VALIDATION ===
    cfg.cv.n_folds = 5;
    cfg.cv.n_repeats = 50;
    cfg.cv.stratified = true;
    
    % === CLASSIFIERS ===
    cfg.classifiers.types = {'LDA', 'PLSDA', 'SVM', 'RandomForest'};
    
    % Hyperparameter grids
    cfg.classifiers.plsda_n_components = 1:10;
    cfg.classifiers.svm_C_values = 10.^(-2:0.5:2);
    cfg.classifiers.svm_gamma_values = 10.^(-3:0.5:1);
    cfg.classifiers.rf_n_trees = [50, 100, 200, 500];
    cfg.classifiers.rf_max_depth = [5, 10, 20, 30];
    
    % === REPRODUCIBILITY ===
    cfg.random_seed = 42;
    
    % === REPORTING ===
    cfg.report.figures_format = 'png';
    cfg.report.figures_dpi = 300;
    cfg.report.save_intermediate = true;
end