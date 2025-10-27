# API Reference - Meningioma FTIR Classification Pipeline

**Version 4.0** | October 2025

Complete reference for all pipeline functions.

---

## Core Functions

### run_pipeline

Main entry point for the complete machine learning pipeline.

**Syntax:**
```matlab
run_pipeline()
run_pipeline('Name', Value, ...)
results = run_pipeline(...)
```

**Options:**
- `'RunEDA'` - Run EDA step (default: `true`)
- `'OutlierMethod'` - `'eda'`, `'qc'`, or `'none'` (default: `'eda'`)
- `'Classifiers'` - Cell array of classifiers (default: `{'LDA','PLSDA','SVM','RandomForest'}`)
- `'NFolds'` - Number of CV folds (default: from config)
- `'NRepeats'` - Number of CV repeats (default: from config)
- `'Verbose'` - Display output (default: `true`)
- `'SaveResults'` - Save to disk (default: `true`)

**Returns:**
```matlab
results.cv_results      % Cross-validation results
results.data            % Loaded data structure
results.config          % Configuration used
results.timestamp       % Run timestamp
```

**Examples:**
```matlab
% Basic usage
run_pipeline()

% Quick test
run_pipeline('NFolds', 3, 'NRepeats', 10)

% Single classifier
run_pipeline('Classifiers', {'SVM'})

% Use QC instead of EDA
run_pipeline('OutlierMethod', 'qc', 'RunEDA', false)
```

---

### run_eda

Run exploratory data analysis with outlier detection.

**Syntax:**
```matlab
run_eda()
run_eda('Name', Value, ...)
eda_results = run_eda(...)
```

**Options:**
- `'PreprocessingType'` - `'PP1'`, `'PP2'`, etc. (default: `'PP1'`)
- `'Verbose'` - Display output (default: `true`)
- `'CreatePlots'` - Generate plots (default: `true`)
- `'TrainDataFile'` - Path to training data (default: `'data/data_table_train.mat'`)

**Returns:**
```matlab
eda_results.pca              % PCA model and outlier flags
eda_results.X_mean           % Mean spectrum
eda_results.wavenumbers      % Wavenumber values
eda_results.probe_ids_pca    % ProbeUID mapping
eda_results.is_train         % Training indicator
```

**Examples:**
```matlab
% Standard usage
run_eda()

% No plots (faster)
run_eda('CreatePlots', false)

% Different preprocessing
run_eda('PreprocessingType', 'PP2')
```

**Output Files:**
- `results/eda/eda_results_PP1.mat` - Main results
- `results/eda/*.png` - Visualization plots

---

### load_pipeline_data

Unified data loader with flexible outlier filtering.

**Syntax:**
```matlab
data = load_pipeline_data(cfg)
data = load_pipeline_data(cfg, 'OutlierMethod', method)
```

**Inputs:**
- `cfg` - Configuration structure from `config()`

**Options:**
- `'OutlierMethod'` - `'eda'`, `'qc'`, or `'none'` (default: `'eda'`)
- `'Verbose'` - Display output (default: `true`)

**Returns:**
```matlab
data.train              % Training set structure
  .spectra              % Cell array {n_samples x 1}
  .labels               % WHO grades (1 or 3)
  .diss_id              % Sample IDs
  .patient_id           % Patient IDs
  .n_samples            % Number of samples
  .total_spectra        % Total spectra count
  .metadata             % Age, sex, etc.

data.test               % Test set (same structure)
data.wavenumbers        % Wavenumber values
data.pca_model          % PCA model (if using EDA)
  .coeff                % PCA coefficients
  .n_comp               % Number of components
  .X_mean               % Mean spectrum
  .explained            % Variance explained
```

**Examples:**
```matlab
cfg = config();

% Use EDA outliers (default)
data = load_pipeline_data(cfg);

% Use QC filtering
data = load_pipeline_data(cfg, 'OutlierMethod', 'qc');

% No filtering
data = load_pipeline_data(cfg, 'OutlierMethod', 'none');
```

---

### run_patientwise_cv_direct

Patient-stratified cross-validation (internal function, called by `run_pipeline`).

**Syntax:**
```matlab
cv_results = run_patientwise_cv_direct(data, cfg)
```

**Inputs:**
- `data` - Data structure from `load_pipeline_data()`
- `cfg` - Configuration structure

**Returns:**
```matlab
cv_results.(clf_name)        % Results for each classifier
  .metrics                   % Performance metrics
    .accuracy_mean/std
    .sensitivity_mean/std
    .specificity_mean/std
    .precision_mean/std
    .f1_mean/std
    .auc_mean/std
  .sample_predictions        % Predicted labels
  .sample_true               % True labels
  .sample_ids                % Sample IDs
  .patient_ids               % Patient IDs
```

---

## Utility Functions

### export_cv_results

Export cross-validation results to files.

**Syntax:**
```matlab
export_cv_results(cv_results, results_dir)
export_cv_results(cv_results, results_dir, 'Pipeline', description)
```

**Inputs:**
- `cv_results` - CV results structure
- `results_dir` - Output directory path

**Options:**
- `'Pipeline'` - Pipeline description (default: `'EDA-based outlier removal'`)
- `'Verbose'` - Display progress (default: `true`)

**Output Files:**
- `cv_summary.txt` - Text summary with metrics
- `cv_predictions.xlsx` - Excel file with predictions

**Example:**
```matlab
export_cv_results(cv_results, 'results/', ...
                 'Pipeline', 'Custom preprocessing');
```

---

### config

Load pipeline configuration.

**Syntax:**
```matlab
cfg = config()
```

**Returns:**
Configuration structure with fields:
- `paths` - Data, model, and results paths
- `qc` - Quality control thresholds
- `pca` - PCA settings
- `cv` - Cross-validation parameters
- `optimization` - Hyperparameter optimization settings
- `classifiers` - Classifier configurations
- `random_seed` - Reproducibility seed

**Example:**
```matlab
cfg = config();

% Modify settings
cfg.cv.n_folds = 10;
cfg.cv.n_repeats = 100;

% Use modified config
data = load_pipeline_data(cfg);
```

---

## Deprecated Functions

These functions still work but display deprecation warnings. Use the new equivalents instead.

### run_pipeline_direct (deprecated)

**Old:**
```matlab
run_pipeline_direct(true)
```

**New:**
```matlab
run_pipeline('OutlierMethod', 'qc')
```

---

### run_pipeline_with_eda (deprecated)

**Old:**
```matlab
run_full_eda()
run_pipeline_with_eda()
```

**New:**
```matlab
run_pipeline()  % EDA is automatic
```

---

### load_data_direct (deprecated)

**Old:**
```matlab
data = load_data_direct(cfg)
```

**New:**
```matlab
data = load_pipeline_data(cfg, 'OutlierMethod', 'qc')
```

---

### load_data_with_eda (deprecated)

**Old:**
```matlab
data = load_data_with_eda(cfg)
```

**New:**
```matlab
data = load_pipeline_data(cfg, 'OutlierMethod', 'eda')
```

---

## Data Structures

### Training/Test Data Table Format

Required columns in `data_table_train.mat` and `data_table_test.mat`:

- `ProbeUID` - Unique probe identifier (numeric)
- `Diss_ID` - Dissection ID (string/cell)
- `Patient_ID` - Patient identifier (string/categorical)
- `WHO_Grade` - WHO classification ('WHO-1' or 'WHO-3')
- `CombinedSpectra` - Cell array containing spectra matrix [n_spectra × n_wavenumbers]
- `Age` - Patient age (numeric)
- `Sex` - Patient sex (categorical: 'M' or 'F')

### Wavenumbers File Format

`wavenumbers.mat` contains:
- `wavenumbers_roi` - Vector of wavenumber values [1 × n_wavenumbers]

---

## Configuration Parameters

### Key Parameters in config.m

**Cross-Validation:**
```matlab
cfg.cv.n_folds = 5;        % K-fold CV
cfg.cv.n_repeats = 50;     % Repeat CV
cfg.cv.stratified = true;  % Stratify by class
```

**PCA (for LDA):**
```matlab
cfg.pca.variance_threshold = 0.95;  % 95% variance
cfg.pca.max_components = 15;        % Max PCs
```

**Hyperparameter Optimization:**
```matlab
cfg.optimization.enabled = true;           % Enable/disable
cfg.optimization.max_evaluations = 30;     % Bayesian iterations
cfg.optimization.use_parallel = true;      % Use parallel pool
```

**Quality Control:**
```matlab
cfg.qc.snr_threshold = 10;              % Min SNR
cfg.qc.max_absorbance = 1.8;            % Max absorbance
cfg.qc.outlier_confidence = 0.99;       % Chi-squared confidence
```

**Classifiers:**
```matlab
cfg.classifiers.types = {'LDA', 'PLSDA', 'SVM', 'RandomForest'};
cfg.classifiers.cost_sensitive = true;     % Cost-sensitive learning
cfg.classifiers.cost_who3_penalty = 5;     % Penalty for missing WHO-3
```

---

## Error Codes

### Common Errors and Solutions

**`FileNotFound`**
- **Cause**: Missing data files
- **Solution**: Run `split_train_test()` to generate data tables

**`EDANotFound`**
- **Cause**: Using `OutlierMethod='eda'` without running EDA
- **Solution**: Run `run_eda()` first or use `'OutlierMethod', 'none'`

**`InvalidConfig`**
- **Cause**: cfg structure missing required fields
- **Solution**: Use `cfg = config()` to get valid configuration

**`OutOfMemory`**
- **Cause**: Insufficient RAM for large datasets
- **Solution**: Reduce `n_repeats` or test fewer classifiers

---

## Performance Optimization

### Speed Improvements

```matlab
% Reduce CV iterations
run_pipeline('NFolds', 3, 'NRepeats', 10)  % ~5-10 min instead of ~30 min

% Test fewer classifiers
run_pipeline('Classifiers', {'SVM'})

% Disable hyperparameter optimization
cfg = config();
cfg.optimization.enabled = false;

% Skip EDA on subsequent runs
run_pipeline('RunEDA', false)
```

### Memory Optimization

```matlab
% Process one classifier at a time
for clf = {'LDA', 'PLSDA', 'SVM', 'RandomForest'}
    results = run_pipeline('Classifiers', clf, 'RunEDA', false);
    % Save results
    save(sprintf('results_%s.mat', clf{1}), 'results');
    clear results;  % Free memory
end
```

---

## Version History

**v4.0** (October 24, 2025)
- Unified pipeline with single entry point
- Consolidated data loaders
- Improved documentation
- Better code organization

**v3.0** (October 21, 2025)
- Direct table access
- Patient-wise stratification
- EDA integration

**v2.0** (Early 2025)
- Patient-wise CV implementation
- Dual-level metrics

**v1.0** (2024)
- Initial pipeline with averaging

---

**Last Updated**: October 24, 2025  
**Version**: 4.0  
**Maintainer**: FTIR Meningioma Classification Team
