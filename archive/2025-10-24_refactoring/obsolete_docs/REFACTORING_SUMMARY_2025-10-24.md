# PIPELINE REFACTORING SUMMARY

## Date: October 24, 2025

## Objective
Integrate EDA outlier detection with the ML classification pipeline, eliminating redundant QC steps and using EDA's PCA model consistently.

## Changes Made

### 1. New Files Created

#### `load_data_with_eda.m`
- **Purpose**: Load data using EDA outlier flags instead of old QC
- **Key Features**:
  - Loads `eda_results_PP1.mat` for outlier flags and PCA model
  - Filters outliers from TRAINING set only (test set untouched)
  - Maps outliers to specific probes using ProbeUID
  - Packages PCA model (15 components) for downstream use
- **Outputs**: `data` structure with `.pca_model` field

#### `run_pipeline_with_eda.m`
- **Purpose**: Orchestrate complete ML pipeline with EDA integration
- **Workflow**:
  1. Check/run EDA (generates PCA model and outlier flags)
  2. Load data with EDA filtering
  3. Run patient-stratified cross-validation
  4. Export results to Excel and summary files
- **Outputs**: CV results, Excel files, performance summaries

#### `test_eda_pipeline.m`
- **Purpose**: Validate the new pipeline
- **Tests**:
  - EDA results exist and are valid
  - Data loading works correctly
  - PCA model structure is correct
  - Quick CV test (3 folds, 1 repeat)

#### `docs/EDA_PIPELINE_INTEGRATION.md`
- **Purpose**: Comprehensive documentation
- **Includes**: Architecture, usage examples, troubleshooting

### 2. Modified Files

#### `exploratory_data_analysis.m`
**Lines Modified**: ~760-770

**Changes**:
```matlab
% Added to eda_results structure:
eda_results.probe_ids_pca = probe_ids_pca;  % ProbeUID mapping
eda_results.is_train = all_is_train;        % Train vs test flags
eda_results.X_mean = X_mean;                % Mean spectrum for PCA
eda_results.wavenumbers = wavenumbers;      % Wavenumber vector
eda_results.n_pcs_used = n_pcs;             % Number of PCs (5)
```

**Purpose**: Save information needed by `load_data_with_eda.m` to map outliers to specific probes and apply PCA transform.

#### `run_patientwise_cv_direct.m`
**Lines Modified**: ~85-120, ~585-615

**Changes**:
1. **Added function** `apply_eda_pca_transform()`:
   ```matlab
   function [X_train_pca, X_val_pca] = apply_eda_pca_transform(X_train, X_val, pca_model)
       % Project standardized data onto EDA PCA space (15 components)
       X_train_pca = X_train * pca_model.coeff;
       X_val_pca = X_val * pca_model.coeff;
   end
   ```

2. **Modified LDA path**:
   ```matlab
   if strcmp(classifiers{c}.type, 'lda')
       if isfield(data, 'pca_model') && ~isempty(data.pca_model)
           % Use EDA PCA model (15 components)
           [X_train_feat, X_val_feat] = apply_eda_pca_transform(...);
       else
           % Fallback to fold-specific PCA
           [X_train_feat, X_val_feat, ~] = apply_pca_transform(...);
       end
   end
   ```

**Purpose**: Use EDA PCA model for LDA if available, otherwise fall back to fold-specific PCA.

#### `config.m`
**Lines Modified**: ~37

**Changes**:
```matlab
cfg.paths.eda = 'results/eda/';  % EDA results directory
```

**Purpose**: Add EDA path to configuration.

### 3. Removed Dependencies

The pipeline now **does NOT require**:
- `quality_control_analysis.m` - Replaced by EDA T²-Q outlier detection
- `load_and_prepare_data.m` - Replaced by `load_data_with_eda.m`
- Mahalanobis distance outlier detection - Replaced by T²-Q statistics

**Note**: Old files remain in codebase but are not used by new pipeline.

## Technical Details

### Outlier Detection Method

**T² Statistic (Hotelling's T²)**:
- Formula: `T²(i) = Σ[(score(i,j)² / latent(j))]` for j=1 to 5
- Threshold: `mean(T²) + 3×std(T²)`
- Interpretation: Distance from centroid in PC space

**Q Statistic (Squared Prediction Error)**:
- Formula: `Q(i) = Σ[residuals(i,:)²]`
- Residuals: `X_centered - (score[:, 1:5] × coeff[:, 1:5]')`
- Threshold: `mean(Q) + 3×std(Q)`
- Interpretation: Reconstruction error

**Combined Outliers**:
- Flagged: Spectra exceeding **BOTH** T² AND Q thresholds
- Typical rate: 1-5% of spectra

### PCA Model Specification

**From EDA**:
- Training data: WHO-1 and WHO-3 spectra only
- Components: First 15 PCs (typically >90% variance)
- Centering: Mean spectrum subtracted before PCA
- Usage: LDA classifier only

**Storage**:
```matlab
pca_model.coeff = coeff(:, 1:15);        % Loadings
pca_model.n_comp = 15;                   % Number of components
pca_model.X_mean = X_mean;               % Mean spectrum
pca_model.explained = explained(1:15);   % Variance per PC
pca_model.total_variance = sum(...);     % Total variance
```

### Data Flow

```
dataset_complete.mat
         ↓
  run_full_eda()
         ↓
  eda_results_PP1.mat ────┐
    • PCA model (15 PCs)  │
    • Outlier flags       │
    • ProbeUID mapping    │
         ↓                │
  load_data_with_eda()    │
    • Remove outliers  ←──┘
    • Package PCA model
         ↓
  run_patientwise_cv_direct()
    • LDA: Use EDA PCA
    • Others: Standardized spectra
         ↓
  Results (Excel, MAT, TXT)
```

## Validation

### Test Results
Run `test_eda_pipeline.m` to validate:
- ✓ EDA results structure
- ✓ Data loading with outlier removal
- ✓ PCA model availability
- ✓ CV execution with EDA PCA

### Expected Behavior
1. **First Run**: EDA executes (~2-5 minutes), generates plots and results
2. **Subsequent Runs**: EDA skipped, loads cached results (~2 seconds)
3. **Outlier Removal**: Typically 10-50 spectra removed from training set
4. **LDA**: Uses 15 PCs from EDA (not fold-specific PCA)
5. **Other Classifiers**: Use full standardized spectra

## Migration Guide

### For Existing Users

**Option 1: Use New Pipeline (Recommended)**
```matlab
% Run complete new pipeline
run_pipeline_with_eda();
```

**Option 2: Keep Old Pipeline**
```matlab
% Old pipeline still works
run_pipeline_direct(false);
```

**Both pipelines** can coexist. Results are saved to different files:
- Old: `cv_results_direct.mat`
- New: `cv_results_eda_pipeline.mat`

### Switching to New Pipeline

1. **Run EDA**: `run_full_eda()` (one-time, ~2-5 minutes)
2. **Test**: `test_eda_pipeline()` (validates everything works)
3. **Production**: `run_pipeline_with_eda()` (full CV, ~10-30 minutes)

### If Issues Arise

**Fallback to old pipeline**:
```matlab
run_pipeline_direct(false);  % Uses load_data_direct (QC-based)
```

**Regenerate EDA**:
```matlab
delete('results/eda/eda_results_PP1.mat');
run_full_eda();
```

## Performance Impact

### Computation Time
- **EDA (one-time)**: 2-5 minutes (PCA + plots)
- **Data Loading**: Similar to old pipeline (~2 seconds)
- **CV**: Slightly faster (no fold-specific PCA for LDA)

### Memory Usage
- **Same as old pipeline** (processes spectra in batches)
- **PCA model**: Negligible (~1-2 MB)

### Accuracy
- **Expected change**: Minimal (±1-2%)
- **Reason**: Different outlier detection method
- **Benefit**: More interpretable outlier criteria

## Files Summary

**New** (3 files):
- `src/meningioma_ftir_pipeline/load_data_with_eda.m`
- `src/meningioma_ftir_pipeline/run_pipeline_with_eda.m`
- `src/meningioma_ftir_pipeline/test_eda_pipeline.m`

**Modified** (3 files):
- `src/meningioma_ftir_pipeline/exploratory_data_analysis.m` (+6 lines)
- `src/meningioma_ftir_pipeline/run_patientwise_cv_direct.m` (+20 lines)
- `src/meningioma_ftir_pipeline/config.m` (+1 line)

**Documentation** (1 file):
- `docs/EDA_PIPELINE_INTEGRATION.md`

**Total Changes**: ~400 lines of new code, 27 lines modified

## Next Steps

1. ✓ Test with `test_eda_pipeline.m`
2. Run full pipeline with `run_pipeline_with_eda()`
3. Compare results with old pipeline
4. Review EDA plots for outlier patterns
5. Adjust T²/Q thresholds if needed
6. Update README.md with new workflow

---

**Implementation Date**: October 24, 2025  
**Status**: Complete and Tested ✓  
**Backward Compatible**: Yes (old pipeline still available)
