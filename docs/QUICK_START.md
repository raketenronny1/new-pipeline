# Quick Start Guide: Patient-Wise Cross-Validation

## Overview

This guide helps you quickly get started with the patient-wise cross-validation implementation for FTIR-based meningioma classification.

> **📝 Note on Numbers**: Throughout this documentation, example numbers like "~768 spectra per patient" or "~44 patients" are based on typical FTIR datasets. The implementation **automatically adapts to your actual data** - whatever number of patients and spectra you have in your `.mat` files.

## Prerequisites

- MATLAB R2020a or later
- Statistics and Machine Learning Toolbox
- Data files in `data/` directory:
  - `data_table_train.mat`
  - `data_table_test.mat`
  - `wavenumbers.mat`

## Installation

1. Navigate to project directory:
```matlab
cd 'c:\Users\Franz\OneDrive\01_Promotion\01 Data\new-pipeline'
```

2. Add source to MATLAB path:
```matlab
addpath('src/meningioma_ftir_pipeline');
```

## Quick Start (3 Steps)

### Option 1: Run Complete Pipeline

```matlab
% Run everything (QC, data loading, PCA, CV, visualization)
run_full_pipeline_patientwise();
```

That's it! The pipeline will:
- ✓ Load patient data (NO averaging)
- ✓ Perform 5-fold patient-wise stratified CV
- ✓ Train classifier on individual spectra
- ✓ Aggregate predictions via majority voting
- ✓ Calculate dual-level metrics
- ✓ Export results to Excel and generate figures

### Option 2: Step-by-Step Control

```matlab
% Step 1: Load configuration
cfg = config();

% Step 2: Load patient-wise data (NO AVERAGING!)
load_and_prepare_data_patientwise(cfg);

% Step 3: Run cross-validation
cvResults = run_patientwise_cross_validation(cfg);

% Step 4: Visualize and export
load('results/meningioma_ftir_pipeline/patientwise_data.mat', 'trainingData');
visualizePatientConfidence(cvResults, cfg.paths.results);
exportDetailedResults(cvResults, trainingData.patientData, cfg.paths.results);
```

## Validation Test

Before running the full pipeline, test the implementation:

```matlab
test_patientwise_implementation();
```

This runs 6 validation checks to ensure everything is set up correctly.

## Configuration

### Basic Settings

Edit `src/meningioma_ftir_pipeline/config.m`:

```matlab
% Number of CV folds
cfg.cv.n_folds = 5;  % Change to 10 for more robust estimates

% Random seed for reproducibility
cfg.random_seed = 42;

% Classifier type
cfg.classifiers.primary_type = 'SVM';  % or 'LDA', 'RandomForest'
```

### Advanced Settings

```matlab
% PCA variance threshold
cfg.pca.variance_threshold = 0.95;  % Keep 95% variance

% Maximum number of principal components
cfg.pca.max_components = 15;

% SVM hyperparameters
cfg.classifiers.svm_C = 1;
cfg.classifiers.svm_gamma = 'auto';
```

## Understanding Results

### Console Output

```
=== FINAL CROSS-VALIDATION RESULTS (5-Fold) ===
Patient-Level Performance (Mean ± SD) [95% CI]:
  Accuracy:    85.00% ± 5.00% [±4.38%]
  Sensitivity: 80.00% ± 8.94% [±7.84%]
  Specificity: 90.00% ± 7.07% [±6.20%]
```

### Output Files

All files saved in `results/meningioma_ftir_pipeline/`:

1. **`cv_results_patientwise.xlsx`** - Detailed predictions per patient
   - Open in Excel for clinical review
   - Check `InterpretationFlag` column for confidence

2. **`cv_results_patientwise_summary.txt`** - Summary statistics
   - Text file with performance metrics
   - Confidence distribution breakdown

3. **`patient_confidence_analysis.png`** - Visualizations
   - 6-panel figure showing:
     - Confidence distribution
     - Correctness vs confidence
     - Uncertainty metrics
     - Confusion matrix

4. **`cv_results_patientwise.mat`** - Full MATLAB results
   - Load for detailed analysis:
     ```matlab
     load('results/meningioma_ftir_pipeline/cv_results_patientwise.mat');
     ```

## Interpreting Clinical Flags

In the Excel file, each patient has an interpretation flag:

- **High Confidence - Correct**: >85% spectra agree, correct prediction ✅
- **High Confidence - INCORRECT**: >85% spectra agree, wrong prediction ⚠️ **REVIEW!**
- **Low Confidence - Ambiguous**: <60% spectra agree (uncertain) ⚡
- **Moderate Confidence**: 60-85% agreement

### Action Items

1. **High Confidence INCORRECT** cases → Manual expert review required
2. **Low Confidence** cases → May need additional diagnostic tests
3. **High Confidence Correct** cases → Most reliable predictions

## Common Tasks

### Change Number of Folds

```matlab
cfg = config();
cfg.cv.n_folds = 10;  % Use 10 folds
cvResults = run_patientwise_cross_validation(cfg);
```

### Use Different Classifier

```matlab
cfg = config();
cfg.classifiers.primary_type = 'RandomForest';
cvResults = run_patientwise_cross_validation(cfg);
```

### Skip Quality Control

```matlab
run_full_pipeline_patientwise(false);  % Skip QC
```

### Skip PCA

```matlab
run_full_pipeline_patientwise(true, false);  % QC=true, PCA=false
```

## Verifying No Data Leakage

The implementation prevents data leakage by:

1. ✅ Creating CV folds at **patient level**
2. ✅ Never splitting patients across folds
3. ✅ Stratifying to maintain class balance

To manually verify:

```matlab
cfg = config();
load('results/meningioma_ftir_pipeline/patientwise_data.mat', 'trainingData');
cvFolds = createPatientWiseStratifiedCV(trainingData.patientData, 5, 42);

% Check fold 1
fold1_train = cvFolds(1).trainPatientIdx;
fold1_test = cvFolds(1).testPatientIdx;

% Should be empty (no overlap)
overlap = intersect(fold1_train, fold1_test);
assert(isempty(overlap), 'Data leakage detected!');
```

## Verifying No Averaging

The patient-wise implementation:

1. ✅ Loads all spectra per patient (typically hundreds)
2. ✅ Trains classifier on individual spectra
3. ✅ Predicts each spectrum independently
4. ✅ Aggregates via majority vote (AFTER prediction)

To verify:

```matlab
load('results/meningioma_ftir_pipeline/patientwise_data.mat', 'trainingData');

% Check first patient - should have many spectra (e.g., 768), NOT 1!
patient1 = trainingData.patientData(1);
fprintf('Patient 1 has %d individual spectra\n', size(patient1.spectra, 1));
% Output example: "Patient 1 has 768 individual spectra"
```

**Note**: The exact number of spectra per patient depends on your data acquisition protocol.

## Performance Expectations

Typical results for meningioma classification:

- **Accuracy**: 80-95% (patient-level)
- **Sensitivity**: 75-90% (WHO-3 detection)
- **Specificity**: 80-95% (WHO-1 detection)
- **CV Runtime**: 5-15 minutes (5 folds, dataset dependent)

**Note**: Numbers used in documentation (e.g., "~32 patients", "~768 spectra") are **examples** based on typical FTIR datasets. The actual implementation automatically adapts to your specific dataset size.

Lower performance compared to spectrum-level metrics is **expected and correct**:
- Spectrum-level: Optimistic (inflated by within-patient correlation)
- Patient-level: Realistic (proper independent test)

## Troubleshooting

### "Patient-wise data not found"

**Solution**: Run data loading first
```matlab
cfg = config();
load_and_prepare_data_patientwise(cfg);
```

### "QC results not found"

**Solution**: Skip QC or run it first
```matlab
run_full_pipeline_patientwise(false);  % Skip QC
% OR
quality_control_analysis(config());    % Run QC first
```

### Memory Issues

**Solution**: Reduce PCA components
```matlab
cfg = config();
cfg.pca.max_components = 10;  % Reduce from 15
```

### Different Results Each Run

**Solution**: Check random seed is set
```matlab
cfg = config();
fprintf('Random seed: %d\n', cfg.random_seed);  % Should be 42
```

## Next Steps

After successful validation:

1. **Examine high-confidence incorrect cases** in Excel file
2. **Adjust hyperparameters** if needed (in `config.m`)
3. **Run with more folds** (e.g., 10-fold) for robust estimates
4. **Compare different classifiers** (SVM, LDA, RandomForest)
5. **Document results** for publication/clinical use

## Getting Help

- See `PATIENT_WISE_CV_README.md` for detailed documentation
- Check function comments in source files
- Run validation test: `test_patientwise_implementation()`
- Review `.github/copilot-instructions.md` for implementation details

## Citation

If using this implementation, please cite:

- Baker et al. (2014) Nature Protocols 9(8):1771-1791
- Greener et al. (2022) Nature Reviews Molecular Cell Biology 23:40-55

---

**Ready to start? Run:**

```matlab
addpath('src/meningioma_ftir_pipeline');
test_patientwise_implementation();  % Validate setup
run_full_pipeline_patientwise();    % Run full pipeline
```
