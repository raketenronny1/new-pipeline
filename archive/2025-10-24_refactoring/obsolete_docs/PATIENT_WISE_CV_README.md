# Patient-Wise Cross-Validation Implementation

## Overview

This implementation refactors the MATLAB codebase for FTIR spectroscopy-based brain tumor classification (WHO Grade 1 vs WHO Grade 3) to use **patient-wise stratified cross-validation** without averaging spectra.

## 🎯 Key Features

### ✅ No Data Leakage
- All ~768 spectra from one patient stay together in the same fold
- Cross-validation performed at **PATIENT LEVEL**, not spectrum level
- Patients never split across train/test sets within a fold

### ✅ No Spectrum Averaging
- Each of the ~768 spectra per patient processed individually
- Predictions made for each spectrum independently
- Aggregation happens ONLY at prediction stage via majority voting

### ✅ Dual-Level Evaluation
- **Spectrum-level**: Individual predictions (supplementary metric)
- **Patient-level**: Majority vote decision (PRIMARY METRIC)

## 📁 New Files Created

### Core Functions

1. **`load_and_prepare_data_patientwise.m`**
   - Loads raw data and creates patient-indexed structure
   - Preserves all spectra per patient (NO AVERAGING)
   - Includes validation checks for data integrity

2. **`patientwise_cv_functions.m`**
   - `createPatientWiseStratifiedCV()`: Creates K-fold splits at patient level
   - `extractSpectraForFold()`: Extracts all spectra for train/test
   - `aggregatePredictionsPerPatient()`: Majority voting implementation

3. **`run_patientwise_cross_validation.m`**
   - Main CV runner with training/prediction loop
   - Preprocessing and PCA application
   - Classifier training on individual spectra

4. **`patientwise_metrics.m`**
   - `computeMetrics()`: Dual-level metric calculation
   - `displayFoldResults()`: Fold-wise result display
   - `aggregateCVResults()`: Across-fold aggregation

5. **`patientwise_visualization.m`**
   - `visualizePatientConfidence()`: 6-panel visualization
   - `exportDetailedResults()`: Excel and text export

6. **`run_full_pipeline_patientwise.m`**
   - Complete pipeline wrapper
   - Quality control integration
   - PCA feature selection
   - Validation checklist

## 🚀 Usage

### Quick Start

```matlab
% Add source to path
addpath('src/meningioma_ftir_pipeline');

% Run complete pipeline
run_full_pipeline_patientwise();
```

### Step-by-Step Execution

```matlab
% 1. Load configuration
cfg = config();

% 2. (Optional) Run quality control
quality_control_analysis(cfg);

% 3. Load patient-wise data (NO AVERAGING)
load_and_prepare_data_patientwise(cfg);

% 4. Run patient-wise cross-validation
cvResults = run_patientwise_cross_validation(cfg);

% 5. Visualize and export results
load('results/meningioma_ftir_pipeline/patientwise_data.mat', 'trainingData');
visualizePatientConfidence(cvResults, cfg.paths.results);
exportDetailedResults(cvResults, trainingData.patientData, cfg.paths.results);
```

### Configuration Options

Update `config.m` to customize:

```matlab
% Cross-validation settings
cfg.cv.n_folds = 5;           % Number of folds (default: 5)
cfg.random_seed = 42;          % Reproducibility seed

% PCA settings
cfg.pca.variance_threshold = 0.95;  % Keep 95% variance
cfg.pca.max_components = 15;        % Upper limit on PCs

% Classifier settings
cfg.classifiers.primary_type = 'SVM';  % 'SVM', 'LDA', or 'RandomForest'
cfg.classifiers.svm_C = 1;
cfg.classifiers.svm_gamma = 'auto';
```

## 📊 Output Files

After running the pipeline, you'll find:

### Results Directory (`results/meningioma_ftir_pipeline/`)

1. **`cv_results_patientwise.mat`**
   - Complete MATLAB results structure with all CV folds

2. **`cv_results_patientwise.xlsx`**
   - Detailed patient-level predictions
   - Columns: PatientID, TrueLabel, PredictedLabel, Confidence, etc.
   - Clinical interpretation flags

3. **`cv_results_patientwise_summary.txt`**
   - Summary statistics (Mean ± SD, 95% CI)
   - Confusion matrix
   - Confidence distribution

4. **`patient_confidence_analysis.png/.fig`**
   - 6-panel visualization:
     - Confidence distribution
     - Confidence vs correctness
     - Entropy distribution
     - Mean vs variability scatter
     - Agreement distribution
     - Confusion matrix heatmap

5. **`patientwise_data.mat`**
   - Intermediate patient-indexed data structure

## 🔍 Data Structure

### Patient Data Organization

```matlab
patientData(i).patientID     % String: Patient identifier
patientData(i).spectra       % [N_spectra × N_wavenumbers] matrix
patientData(i).label         % Numeric: 1 (WHO-1) or 3 (WHO-3)
patientData(i).probe_id      % String: Sample ID
patientData(i).metadata      % Struct: age, sex, n_spectra, etc.
```

### CV Results Structure

```matlab
cvResults(k).fold                    % Fold number
cvResults(k).spectrumLevelResults    % Individual spectrum predictions
cvResults(k).patientLevelResults     % Aggregated patient predictions
cvResults(k).trainedModel            % Trained classifier
cvResults(k).spectrumMetrics         % Spectrum-level metrics
cvResults(k).patientMetrics          % Patient-level metrics (PRIMARY)
cvResults(k).confidenceMetrics       % Uncertainty quantification

% Aggregated across folds
cvResults(1).aggregated.meanAccuracy     % Mean ± SD
cvResults(1).aggregated.meanSensitivity  % With 95% CI
cvResults(1).aggregated.meanSpecificity
```

## 📈 Metrics Explained

### Patient-Level Metrics (PRIMARY)

- **Accuracy**: Percentage of patients correctly classified
- **Sensitivity**: True positive rate (WHO-3 detection)
- **Specificity**: True negative rate (WHO-1 detection)
- **PPV/NPV**: Positive/Negative predictive values
- **F1-Score**: Harmonic mean of precision and recall

### Confidence Metrics

- **Majority Vote Confidence**: % of spectra agreeing with final prediction
- **Prediction Entropy**: Shannon entropy of probability distribution
- **Std Probability**: Variability across spectrum predictions
- **Agreement %**: Percentage of spectra voting for majority class

### Clinical Interpretation Flags

- **High Confidence - Correct**: >85% agreement, correct prediction
- **High Confidence - INCORRECT**: >85% agreement, wrong prediction (⚠️ REVIEW)
- **Low Confidence - Ambiguous**: <60% agreement (uncertain case)
- **Moderate Confidence**: 60-85% agreement

## ✅ Validation Checklist

The implementation ensures:

- [✓] No data leakage (patients separate in train/test)
- [✓] All spectra preserved (no averaging before prediction)
- [✓] Stratified CV (both classes in each fold)
- [✓] Majority voting implemented per patient
- [✓] Patient-level metrics calculated (primary)
- [✓] Confidence metrics (entropy, std, agreement)
- [✓] Clinical interpretation (high/low confidence flags)
- [✓] Reproducibility (random seed set)
- [✓] Documentation and code comments
- [✓] Output files (Excel, summary, figures)

## 🔬 Scientific References

This implementation follows best practices from:

1. **Baker et al. (2014)** Nature Protocols 9(8):1771-1791
   - "Using Fourier transform IR spectroscopy to analyze biological materials"
   - Guidelines for FTIR data handling and quality control

2. **Greener et al. (2022)** Nature Reviews Molecular Cell Biology 23:40-55
   - "A guide to machine learning for biologists"
   - Best practices for ML in biomedical research
   - Emphasis on preventing data leakage and proper CV

## 🔧 Differences from Original Pipeline

### Original Implementation
- ❌ Averaged ~768 spectra per patient into single representative spectrum
- ❌ CV performed on averaged samples (not patient-wise)
- ❌ No explicit data leakage prevention
- ❌ Single-level metrics only

### New Patient-Wise Implementation
- ✅ Preserves all individual spectra
- ✅ Patient-wise stratified CV (no leakage)
- ✅ Individual spectrum prediction + majority voting
- ✅ Dual-level metrics (spectrum + patient)
- ✅ Confidence quantification for clinical use

## 🐛 Troubleshooting

### Common Issues

**Issue**: "Patient-wise data not found"
```matlab
% Solution: Run data loading first
cfg = config();
load_and_prepare_data_patientwise(cfg);
```

**Issue**: "QC results not found"
```matlab
% Solution: Either run QC or skip it
quality_control_analysis(cfg);  % Run QC
% OR
run_full_pipeline_patientwise(false);  % Skip QC
```

**Issue**: Memory issues with large datasets
```matlab
% Solution: Use v7.3 MAT files (already implemented)
% Or reduce PCA components in config.m
cfg.pca.max_components = 10;
```

## 📝 Example Results Interpretation

### Sample Output
```
=== FINAL CROSS-VALIDATION RESULTS (5-Fold) ===
Patient-Level Performance (Mean ± SD) [95% CI]:
  Accuracy:    85.00% ± 5.00% [±4.38%]
  Sensitivity: 80.00% ± 8.94% [±7.84%]
  Specificity: 90.00% ± 7.07% [±6.20%]
```

### Interpretation
- **Accuracy**: On average, 85% of patients correctly classified
- **95% CI**: True performance likely between 80.62% and 89.38%
- **Sensitivity**: 80% of WHO-3 patients correctly identified
- **Specificity**: 90% of WHO-1 patients correctly identified

### Clinical Utility
- High confidence predictions (>85% agreement) have higher accuracy
- Low confidence cases may require additional diagnostic tests
- Misclassified high-confidence cases warrant manual review

## 📧 Support

For questions or issues, refer to:
- Original pipeline documentation in `src/meningioma_ftir_pipeline/README.md`
- Code comments in individual function files
- Copilot instructions in `.github/copilot-instructions.md`

## 📄 License

See `src/utils/license.txt` for licensing information.
