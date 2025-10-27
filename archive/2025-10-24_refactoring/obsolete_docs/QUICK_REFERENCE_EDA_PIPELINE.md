# QUICK REFERENCE: EDA-INTEGRATED PIPELINE

## One-Line Commands

```matlab
% Run everything (auto-detects if EDA needed)
run_pipeline_with_eda()

% Test pipeline
test_eda_pipeline()

% Run EDA only
run_full_eda()

% Load data with EDA filtering
cfg = config(); data = load_data_with_eda(cfg);
```

## Pipeline Comparison

| Task | Old Pipeline | New Pipeline |
|------|-------------|--------------|
| **Run Complete** | `run_pipeline_direct(false)` | `run_pipeline_with_eda()` |
| **Load Data** | `load_data_direct(cfg)` | `load_data_with_eda(cfg)` |
| **Outlier Detection** | Mahalanobis (in QC) | T²-Q (in EDA) |
| **PCA for LDA** | Per-fold | From EDA (15 PCs) |
| **Results File** | `cv_results_direct.mat` | `cv_results_eda_pipeline.mat` |

## Key Features

✓ **Single PCA Model**: Computed once in EDA, used consistently  
✓ **T²-Q Outliers**: Statistical outlier detection (1-5% flagged)  
✓ **15 PCs for LDA**: Captures >90% variance  
✓ **Raw Features for Others**: PLS-DA, SVM, RF use full spectra  
✓ **Patient Stratification**: No patient leakage in CV  
✓ **Test Set Untouched**: No outlier removal on test data  

## File Locations

**Input**:
- `data/dataset_complete.mat` - Complete dataset
- `data/data_table_train.mat` - Training set
- `data/data_table_test.mat` - Test set

**EDA Output**:
- `results/eda/eda_results_PP1.mat` - PCA model + outliers
- `results/eda/*.png` - Visualizations

**CV Output**:
- `results/meningioma_ftir_pipeline/cv_results_eda_pipeline.mat`
- `results/meningioma_ftir_pipeline/cv_predictions.xlsx`
- `results/meningioma_ftir_pipeline/cv_summary.txt`

## Troubleshooting

| Error | Solution |
|-------|----------|
| "EDA results not found" | Run `run_full_eda()` |
| "Spectrum count mismatch" | Regenerate EDA with current dataset |
| "No samples after filtering" | Check outlier thresholds in EDA |
| Old results loaded | Delete cached files, rerun |

## Performance

**Expected Time**:
- EDA (first run): 2-5 minutes
- Data loading: ~2 seconds
- CV (5 folds, 50 repeats): 10-30 minutes

**Expected Metrics** (with outlier removal):
- LDA: 0.82-0.88
- PLS-DA: 0.85-0.90
- SVM: 0.86-0.92 ⭐ Best
- RF: 0.84-0.89

## Customization

**Adjust Outlier Thresholds** (in `exploratory_data_analysis.m`):
```matlab
% Line ~340-345
t2_threshold = 3;  % Increase to keep more spectra (e.g., 4 or 5)
q_threshold = 3;   % Increase to keep more spectra (e.g., 4 or 5)
```

**Change PCA Components** (in `load_data_with_eda.m`):
```matlab
% Line ~87
n_pcs_to_use = min(15, size(pca_info.coeff, 2));  % Change 15 to desired number
```

**Modify CV Settings** (in `config.m`):
```matlab
cfg.cv.n_folds = 5;      % K-fold CV
cfg.cv.n_repeats = 50;   % Number of repetitions
```

## Workflow

```
┌──────────────────┐
│  EDA (Once)      │ → eda_results_PP1.mat
│  - PCA Model     │   (15 PCs, outlier flags)
│  - Outliers      │
└────────┬─────────┘
         ↓
┌────────────────────────┐
│  Load Data             │ → data structure
│  - Remove outliers     │   (with pca_model)
│  - Package PCA         │
└────────┬───────────────┘
         ↓
┌────────────────────────┐
│  Cross-Validation      │ → cv_results
│  - LDA: Use EDA PCA    │   Excel, MAT, TXT
│  - Others: Raw spectra │
└────────────────────────┘
```

---
**Version**: 3.0 (EDA-Integrated)  
**Date**: 2025-10-24
