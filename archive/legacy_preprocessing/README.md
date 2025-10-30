# Legacy Preprocessing Functions

**Archived:** October 30, 2025  
**Reason:** Superseded by `PreprocessingPipeline` class in modular architecture (v5.0)

## Contents

This directory contains standalone preprocessing functions from the previous pipeline implementation (v4.0 and earlier):

1. **`prepare_ftir_dataset.m`** (595 lines)
   - Creates complete FTIR dataset with dual preprocessing (PP1/PP2)
   - Used in earlier versions to generate `dataset_complete.mat`
   - Superseded by: `prepare_data.m` (data cleaning) + `PreprocessingPipeline` class

2. **`preprocess_spectra.m`** (217 lines)
   - Main orchestrator for spectral preprocessing
   - Standalone function-based approach
   - Superseded by: `PreprocessingPipeline` class with fit/transform pattern

3. **`create_preprocessing_config.m`** (166 lines)
   - Configuration structure generator for PP1/PP2 approaches
   - Superseded by: `PreprocessingPipeline` constructor with BSNCX notation

4. **`test_preprocessing_functions.m`** (280 lines)
   - Test suite for legacy preprocessing functions
   - Uses synthetic data to validate preprocessing steps
   - Superseded by: `tests/test_preprocessing_pipeline.m`

## Why These Were Archived

### Version 4.0 → Version 5.0 Transition

**Old Approach (Function-based):**
```matlab
cfg = create_preprocessing_config('PP1');
[processed, wn] = preprocess_spectra(raw_spectra, wavenumbers, cfg);
```

**New Approach (Object-oriented):**
```matlab
pipeline = PreprocessingPipeline('01020', 'Verbose', true);
[X_train_prep, params] = pipeline.fit_transform(X_train);
X_test_prep = pipeline.transform(X_test, params);
```

### Key Improvements in v5.0

1. **Data Leakage Prevention**: Fit/transform separation ensures preprocessing parameters are learned only from training data
2. **BSNCX Notation**: More flexible configuration (Position 1-5 encoding)
3. **Integration**: Seamless integration with `CrossValidationEngine`
4. **Testing**: Comprehensive unit tests with data leakage verification

## When to Use These Files

These files may still be useful for:

- **Historical reference**: Understanding how preprocessing evolved
- **Data generation**: If you need to regenerate `dataset_complete.mat` with PP1/PP2 columns
- **Validation**: Cross-checking preprocessing results against legacy implementation
- **Documentation**: Examples of FTIR preprocessing steps

## Migration Guide

If you need functionality from these files:

| Legacy Function | Modern Equivalent |
|-----------------|-------------------|
| `prepare_ftir_dataset()` | `prepare_data.m` + `PreprocessingPipeline` |
| `preprocess_spectra(X, wn, 'PP1')` | `pipeline = PreprocessingPipeline('01020'); pipeline.fit_transform(X)` |
| `create_preprocessing_config('PP1')` | `PreprocessingPipeline('01020')` (BSNCX notation) |
| Test suite | `tests/test_preprocessing_pipeline.m` |

## BSNCX vs PP1/PP2 Mapping

**PP1 (Standard):**
- BSNCX: `01020`
- No binning, no smoothing, vector norm, 2nd derivative

**PP2 (Enhanced):**
- BSNCX: `11020`
- Binning (factor 4), SG smoothing, vector norm, 2nd derivative

## See Also

- `src/preprocessing/PreprocessingPipeline.m` - Modern preprocessing class
- `src/preprocessing/README.md` - Current preprocessing documentation
- `tests/test_preprocessing_pipeline.m` - Modern test suite
- `IMPLEMENTATION_SUMMARY.md` - v5.0 architecture details
