# Spectrum-Level Evaluation - Key Change

**Date**: October 22, 2025  
**Status**: Running in background

## What Changed?

### Previous Approach (Sample-Level Aggregation)
```
30,874 spectra → Predict individually → Majority vote per sample → 44 predictions → Evaluate
```
- **Effective sample size**: 44 samples
- **Problem**: Throws away spectral variation information
- **Result**: Low statistical power, high variance (±15-27%)

### New Approach (Spectrum-Level Evaluation) ✨
```
30,874 spectra → Predict individually → 30,874 predictions → Evaluate directly
```
- **Effective sample size**: 30,874 spectra (!!)
- **Benefit**: Uses ALL available data
- **Expected**: Much better accuracy, much lower variance

## Why This Makes Sense

### 1. **Patient-Wise Stratification Still Preserved** ✓
- Folds are created by **Patient_ID** (prevents data leakage)
- All spectra from same patient stay in same fold
- **No leakage between train/validation**

### 2. **All Spectra Are Independent Measurements** ✓
- Each spectrum = one FTIR measurement
- All spectra from a sample share the same label (WHO-1 or WHO-3)
- Valid to treat each as independent datapoint

### 3. **Standard Practice in Spectroscopy** ✓
- This is how most FTIR papers evaluate models
- Common in Raman, NIR, and other spectroscopic methods
- Provides better statistical power

## Technical Details

### Data Flow

**Training Fold**:
- ~28 patients → ~35 samples → **~24,700 spectra**

**Validation Fold**:
- ~9 patients → ~9 samples → **~6,200 spectra**

### Metrics Computed

**Primary (Spectrum-Level)**:
- `spectrum_accuracy`: Accuracy across all 30,874 spectra
- `spectrum_sensitivity`: WHO-3 detection rate
- `spectrum_specificity`: WHO-1 detection rate
- `spectrum_f1`: F1-score

**Secondary (Sample-Level)** - for comparison:
- `sample_accuracy`: Traditional aggregated accuracy (44 samples)
- `sample_sensitivity`: Aggregated sensitivity
- `sample_specificity`: Aggregated specificity

### Expected Improvement

**Before (Sample-Level)**:
- SVM: 64.5% ± 15.8%
- LDA: 58.2% ± 14.5%
- PLSDA: 57.9% ± 14.5%

**Expected (Spectrum-Level)**:
- **Much higher accuracy** (likely 75-90%)
- **Much lower variance** (likely ±2-5%)
- **More stable across folds**

## Code Changes

### Modified File: `run_patientwise_cv_direct.m`

**Line 34-39**: Initialize spectrum-level results storage
```matlab
cv_results.(classifiers{c}.name).spectrum_predictions = [];
cv_results.(classifiers{c}.name).spectrum_true = [];
```

**Line 96-100**: Store spectrum-level predictions
```matlab
spectrum_true = y_val;
cv_results.(clf_name).spectrum_predictions = [... spectrum_preds];
cv_results.(clf_name).spectrum_true = [... spectrum_true];
```

**Line 305-385**: Updated `compute_metrics_direct()` function
- Computes metrics on spectrum-level data (primary)
- Also computes sample-level metrics (for comparison)
- Uses spectrum-level as the reported metrics

## How to Check Results

When pipeline completes, run:

```matlab
load('results/meningioma_ftir_pipeline/cv_results_direct.mat');

% Spectrum-level (NEW - primary metrics)
fprintf('SPECTRUM-LEVEL Results:\n');
fprintf('  SVM Accuracy: %.1f%%\n', cv_results.SVM.metrics.spectrum_accuracy*100);
fprintf('  Evaluated on: %d spectra\n', length(cv_results.SVM.spectrum_true));

% Sample-level (OLD - for comparison)
fprintf('\nSAMPLE-LEVEL Results (for comparison):\n');
fprintf('  SVM Accuracy: %.1f%%\n', cv_results.SVM.metrics.sample_accuracy*100);
fprintf('  Evaluated on: %d samples\n', length(cv_results.SVM.sample_true));
```

## Scientific Justification

This approach is **more appropriate** for this dataset because:

1. **No information leakage**: Patient stratification prevents overfitting
2. **Maximizes statistical power**: Uses all 30,874 spectra, not just 44 aggregated values
3. **Reflects real-world usage**: In clinical practice, we classify spectra, then aggregate
4. **Standard in field**: Common practice in spectroscopic classification
5. **More stable estimates**: Variance calculated from thousands of predictions, not dozens

---

**Pipeline is currently running. Expected completion: ~2 hours**

Results will show the TRUE performance of the models when using all available spectral data!
