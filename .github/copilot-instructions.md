# Meningioma FT-IR Classification Pipeline Guidelines

This document provides essential information for AI coding agents working with this MATLAB-based FTIR spectroscopy analysis pipeline.

## Architecture Overview

The pipeline processes FTIR spectroscopy data from meningioma samples through several sequential phases:

1. **Quality Control**: Filters poor-quality spectra based on SNR, baseline, absorbance metrics (`quality_control_analysis.m`)
2. **Data Preparation**: Creates representative spectra from quality-filtered data (`load_and_prepare_data.m`)
3. **Feature Engineering**: Applies PCA for dimensionality reduction (`perform_feature_selection.m`)
4. **Cross-validation**: Evaluates multiple classifiers with hyperparameter optimization (`run_cross_validation.m`)
5. **Final Model Training**: Trains best model on full training set (`train_final_model.m`)
6. **Test Evaluation**: Applies model to independent test set (`evaluate_test_set.m`)
7. **Report Generation**: Creates performance summaries and visualizations (`generate_report.m`)

## Key Data Structures

- **Input Data Format**: MATLAB tables in `*.mat` files with `CombinedSpectra` cell arrays containing multiple spectra per sample
- **QC Results**: Boolean masks identifying valid spectra and samples in `qc_results` structure
- **Feature Transformed Data**: PCA-transformed features stored in `X_train_pca.mat`
- **Model Objects**: Saved as MATLAB objects in the models directory with preprocessing parameters

## Important Conventions

1. **Configuration Management**: All parameters are centralized in `config.m` and passed as a struct through the pipeline
2. **Directory Structure**:
   ```
   project_root/
   ├── data/                   # Raw data files
   ├── models/                 # Trained models
   ├── results/                # Results and visualizations
   └── src/                    # Source code
       └── meningioma_ftir_pipeline/
   ```
3. **Error Handling**: Use structured error handling with detailed error messages and recovery mechanisms
4. **Fixed vs. Original Functions**: Fixed versions of problematic functions are in the test directory
5. **Categorical Variable Handling**: WHO_Grade needs special handling as categorical data (see `fix_categorical_issues.m`)

## Developer Workflow

### Setup and Running the Pipeline

1. Add source to MATLAB path:
   ```matlab
   addpath('src/meningioma_ftir_pipeline');
   ```

2. Run full pipeline:
   ```matlab
   run_full_pipeline();
   ```

3. Run individual phases:
   ```matlab
   cfg = config();
   load_and_prepare_data(cfg);
   perform_feature_selection(cfg);
   % etc.
   ```

### Debugging

1. Use test directory for debugging with simplified data:
   ```matlab
   cd src/meningioma_ftir_pipeline/test
   run_test_with_debug()
   ```

2. Check results and logs in the results directory - each run creates a timestamped folder

3. For pipeline failures, examine intermediate MAT files in the latest run directory

## Common Issues and Solutions

1. **Categorical Variable Issues**: Convert WHO_Grade to categorical data type before processing
2. **NaN/Inf Values**: Always check for and handle NaN/Inf values, especially after averaging spectra
3. **Path Management**: Use absolute file paths constructed with `fullfile()` for cross-platform compatibility
4. **Data Size Handling**: The pipeline implements memory-efficient processing for large spectral datasets

Remember to use the fixed versions in the test directory when debugging data processing issues, as they include important bug fixes for handling categorical variables and PCA application.