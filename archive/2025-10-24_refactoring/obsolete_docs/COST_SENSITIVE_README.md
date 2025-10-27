# Meningioma FT-IR Classification Pipeline

A MATLAB-based pipeline for classifying meningioma tumor samples (WHO Grade 1 vs Grade 3) using Fourier Transform Infrared (FT-IR) spectroscopy data with cost-sensitive learning to prioritize detection of malignant tumors.

## Overview

This pipeline processes FT-IR spectroscopy data through quality control, feature engineering, and classification with multiple machine learning algorithms. It implements cost-sensitive learning to maximize detection of malignant WHO Grade 3 tumors, which is critical for clinical applications.

## Key Features

- **Quality Control**: Comprehensive filtering based on SNR, absorbance, baseline stability, and correlation metrics
- **Cost-Sensitive Learning**: Prioritizes detection of malignant WHO-3 tumors through:
  - Weighted class priors (LDA)
  - Asymmetric cost matrices (SVM)
  - Sample weighting (Random Forest)
- **Multiple Classifiers**: LDA, PLS-DA, SVM (RBF kernel), Random Forest
- **Hyperparameter Optimization**: Bayesian optimization with parallel processing
- **Spectrum-Level & Sample-Level Evaluation**: Predictions at both granularity levels
- **PCA Only for LDA**: Selective dimensionality reduction based on classifier requirements

## Requirements

- MATLAB R2023b or later
- Statistics and Machine Learning Toolbox
- Parallel Computing Toolbox (optional, for faster optimization)

## Quick Start

```matlab
% 1. Configure pipeline
cfg = config();

% 2. Load and prepare data
data = load_data_direct(cfg);

% 3. Evaluate on test set with cost-sensitive SVM
results = evaluate_test_set_direct(data, cfg, 'SVM');
```

## Pipeline Structure

```
src/meningioma_ftir_pipeline/
├── config.m                      # Central configuration
├── load_data_direct.m            # Data loading with QC
├── evaluate_test_set_direct.m    # Test set evaluation
└── run_patientwise_cv_direct.m   # Cross-validation (patient-wise)

data/
├── data_table_train.mat          # Training data
├── data_table_test.mat           # Test data
└── wavenumbers.mat               # Wavenumber values

results/meningioma_ftir_pipeline/
├── qc/                           # Quality control results
└── test_results_direct.mat       # Final test results
```

## Configuration

Key parameters in `config.m`:

```matlab
% Cost-Sensitive Learning
cfg.classifiers.cost_who3_penalty = 5;  % Penalty for missing WHO-3 (1-10)
                                        % Higher = more aggressive WHO-3 detection
                                        % Recommended: 3-7 for clinical use

% Quality Control
cfg.qc.snr_threshold = 10;              % Minimum SNR
cfg.qc.max_absorbance = 1.8;            % Maximum absorbance

% PCA (LDA only)
cfg.pca.variance_threshold = 0.95;      % Retain 95% variance
cfg.pca.max_components = 15;            % Maximum components

% Hyperparameter Optimization
cfg.optimization.enabled = true;         % Enable Bayesian optimization
cfg.optimization.max_evaluations = 20;   % Optimization iterations
```

## Cost-Sensitive Learning

The pipeline implements cost-sensitive learning to address the clinical requirement of minimizing false negatives for malignant WHO-3 tumors:

### Implementation by Classifier:

- **SVM**: Asymmetric cost matrix `[0, 1; penalty, 0]` penalizes WHO-3→WHO-1 errors
- **Random Forest**: Sample weights (WHO-3 samples weighted by penalty factor)
- **LDA**: Weighted class priors emphasizing WHO-3
- **PLS-DA**: Cost penalty stored for reference (regression-based method)

### Performance with Cost Penalty = 5:

| Classifier | WHO-3 Detection | Overall Accuracy | Notes |
|------------|----------------|------------------|-------|
| SVM        | 90% (18/20)    | 90.6%           | Excellent balance |
| Random Forest | 85% (17/20) | 90.6%           | Good performance |
| PLS-DA     | 80% (16/20)    | 81.2%           | Baseline |
| LDA        | 100% (20/20)   | 62.5%           | Too aggressive |

## Usage Examples

### Basic Test Set Evaluation

```matlab
cfg = config();
data = load_data_direct(cfg);

% Test all classifiers
classifiers = {'LDA', 'PLSDA', 'SVM', 'RandomForest'};
for i = 1:length(classifiers)
    fprintf('\n=== Testing %s ===\n', classifiers{i});
    results = evaluate_test_set_direct(data, cfg, classifiers{i});
end
```

### Adjust Cost Penalty

```matlab
cfg = config();
cfg.classifiers.cost_who3_penalty = 7;  % More aggressive WHO-3 detection
data = load_data_direct(cfg);
results = evaluate_test_set_direct(data, cfg, 'SVM');
```

### Disable Cost-Sensitive Learning

```matlab
cfg = config();
cfg.classifiers.cost_sensitive = false;
cfg.classifiers.cost_who3_penalty = 1;  % No penalty
data = load_data_direct(cfg);
results = evaluate_test_set_direct(data, cfg, 'SVM');
```

## Output Structure

Results from `evaluate_test_set_direct()`:

```matlab
test_results = 
    spectrum_predictions: [N×1 double]      % Predictions (1 or 3)
    spectrum_true: [N×1 double]             % True labels
    spectrum_metrics: struct                % Accuracy, sensitivity, etc.
    
    sample_predictions: [M×1 double]        % Sample-level (majority vote)
    sample_true: [M×1 double]               % True sample labels
    sample_metrics: struct                  % Sample-level metrics
    sample_ids: {M×1 cell}                  % Diss_IDs
    patient_ids: {M×1 cell}                 % Patient_IDs
    
    final_model: [object]                   % Trained model
    pca_model: struct or []                 % PCA model (if LDA)
    std_params: struct                      % Standardization parameters
    classifier_name: char                   % 'SVM', 'LDA', etc.
```

## Performance Metrics

### Spectrum-Level Metrics:
- Predictions on individual spectra (~30,000 training, ~23,000 test)
- Direct classifier output before aggregation

### Sample-Level Metrics:
- Predictions aggregated via majority voting
- Clinically relevant (one prediction per tumor sample)
- **Sensitivity (WHO-1 detection)**: True positive rate for benign tumors
- **Specificity (WHO-3 detection)**: True positive rate for malignant tumors

## Important Notes

1. **PCA is ONLY applied for LDA classifier** - Other classifiers use z-score standardized spectra directly
2. **Cost-sensitive learning affects training only** - Test predictions use the trained cost-sensitive model
3. **Patient-wise stratification** - Cross-validation respects patient boundaries (some patients have multiple samples)
4. **Quality control** - Applied during data loading; rejected spectra/samples are excluded
5. **Majority voting** - Sample-level predictions aggregate multiple spectra per sample

## Troubleshooting

### Too Many False Negatives (Missing WHO-3):
- Increase `cfg.classifiers.cost_who3_penalty` (try 7-10)
- Use SVM or Random Forest (better cost-sensitive performance)

### Too Many False Positives (Over-predicting WHO-3):
- Decrease `cfg.classifiers.cost_who3_penalty` (try 2-3)
- Check if LDA is being used (tends to over-correct)

### Low Overall Accuracy:
- Check QC results (`results/meningioma_ftir_pipeline/qc/`)
- Verify data quality and sample sizes
- Consider adjusting QC thresholds in config

## References

Data structure follows the format established in the integrated QC workflow. Cost-sensitive learning implementations follow MATLAB documentation for:
- `fitcsvm` (Cost parameter)
- `TreeBagger` (Weights parameter)
- `fitcdiscr` (Prior parameter)

## License

Academic/Research Use Only

## Contact

For questions or issues, contact the meningioma FT-IR classification team.

---

**Version**: 1.0  
**Last Updated**: October 2025  
**Status**: Production-ready with cost-sensitive learning
