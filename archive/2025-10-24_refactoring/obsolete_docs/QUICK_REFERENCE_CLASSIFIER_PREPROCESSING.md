# Quick Reference: Classifier Preprocessing Pipeline

## Preprocessing Flow by Classifier

```
RAW SPECTRA
    ↓
┌───────────────────────────────────────┐
│  STANDARDIZATION (ALL CLASSIFIERS)   │
│  - Z-score normalization              │
│  - μ and σ from training data         │
└───────────────────────────────────────┘
    ↓
    ├─────────────┬─────────────┬─────────────┬─────────────┐
    ↓             ↓             ↓             ↓             ↓
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  
│   LDA    │  │  PLS-DA  │  │   SVM    │  │ RF/Ens   │  
└──────────┘  └──────────┘  └──────────┘  └──────────┘  
    ↓             ↓             ↓             ↓
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│   PCA    │  │ ORIGINAL │  │ ORIGINAL │  │ ORIGINAL │
│ ~20-50   │  │  ~900+   │  │  ~900+   │  │  ~900+   │
│  PCs     │  │ features │  │ features │  │ features │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
```

## Classifier Configuration Matrix

| Classifier | Input Features | Kernel/Method | Key Parameters |
|------------|---------------|---------------|----------------|
| **LDA** | PCA components (~20-50) | Linear discriminant | `'DiscrimType', 'linear'` |
| **PLS-DA** | Original spectra (~900+) | Partial least squares | `n_components = 5` |
| **SVM** | Original spectra (~900+) | RBF kernel | `'KernelFunction', 'rbf'`, `'KernelScale', 'auto'` |
| **Random Forest** | Original spectra (~900+) | Ensemble trees | `n_trees = 100` |

## Code Locations

### Cross-Validation: `run_patientwise_cv_direct.m`
```matlab
% Line 67-81: Preprocessing logic
[X_train_std, X_val_std, ~] = standardize_spectra(X_train, X_val);

if strcmp(classifiers{c}.type, 'lda')
    [X_train_feat, X_val_feat, ~] = apply_pca_transform(X_train_std, X_val_std, cfg);
else
    X_train_feat = X_train_std;  % PLS-DA, SVM, RF
    X_val_feat = X_val_std;
end
```

### Test Evaluation: `evaluate_test_set_direct.m`
```matlab
% Line 27-44: Preprocessing logic
[X_train_std, std_params] = standardize_spectra_train(X_train);

if strcmp(classifier_cfg.type, 'lda')
    [X_train_feat, ~, pca_model] = apply_pca_transform_train(X_train_std, cfg);
    use_pca = true;
else
    X_train_feat = X_train_std;  % PLS-DA, SVM, RF
    use_pca = false;
end
```

## QC Rejection Tracking

### Output Files
```
results/meningioma_ftir_pipeline/qc/
├── qc_rejected_spectra_train.csv
└── qc_rejected_spectra_test.csv
```

### CSV Columns
- `Sample_Index`: Index in dataTable
- `Diss_ID`: Sample identifier
- `Patient_ID`: Patient identifier
- `Spectrum_Index`: Which spectrum within sample
- `Rejection_Reason`: 'SNR' | 'Saturation' | 'Baseline' | 'AmideRatio' | 'Mahalanobis'
- `QC_Value`: Actual metric value
- `Sample_WHO_Grade`: Grade of the sample

### Code Location: `quality_control_analysis.m`
```matlab
% Line 129+: Track SNR rejections
rejected_list = track_rejections(rejected_list, snr_failed, i, ...);

% Line 141+: Track saturation rejections
rejected_list = track_rejections(rejected_list, saturation_failed, i, ...);

% Similar for baseline, amide ratio, Mahalanobis
```

## Sample Aggregation

### Method: MAJORITY VOTE
```matlab
function sample_preds = aggregate_to_samples(spectrum_preds, sample_map, n_samples)
    sample_preds = zeros(n_samples, 1);
    for s = 1:n_samples
        sample_preds(s) = mode(spectrum_preds(sample_map == s));  % Most common
    end
end
```

### Documentation in Results
```matlab
% Cross-validation results
cv_results.(clf_name).aggregation_method = 'majority_vote';

% Test set results
test_results.aggregation_method = 'majority_vote';
```

## Validation Checklist

Before running pipeline:
- [ ] Check config.m for PCA parameters (variance_threshold, max_components)
- [ ] Verify QC thresholds appropriate for dataset
- [ ] Ensure correct classifier selected for test evaluation

After running:
- [ ] Verify qc_rejected_spectra_*.csv files created
- [ ] Check that LDA model has ~20-50 features (PCs)
- [ ] Check that other models have ~900+ features (original spectra)
- [ ] Confirm aggregation_method = 'majority_vote' in results

## Common Issues

### Issue: All classifiers have same feature dimension
**Cause**: PCA applied to all classifiers  
**Fix**: Verify conditional PCA logic in lines 73-81 of `run_patientwise_cv_direct.m`

### Issue: SVM performance poor
**Possible causes**:
- Not using RBF kernel
- Not standardizing input
- Using PCA-reduced features instead of full spectra

**Fix**: Check SVM configuration (line 303 in `run_patientwise_cv_direct.m`)

### Issue: QC rejection log empty
**Cause**: No spectra failed QC  
**Check**: Verify QC thresholds not too lenient

## Performance Expectations

### Expected Changes After Fix:
- **LDA**: Similar performance (no change in preprocessing)
- **PLS-DA**: Improved (now uses full spectral information)
- **SVM**: Significantly improved (full spectra + RBF kernel)
- **Random Forest**: Improved (full spectra)

### Feature Dimensions:
- **Before fix**: All ~20-50 (all used PCA)
- **After fix**:
  - LDA: ~20-50 (PCA)
  - Others: ~900+ (original)
