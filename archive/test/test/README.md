# MENINGIOMA FTIR PIPELINE TEST SUITE

## Overview
This directory contains test scripts and data for validating the meningioma FTIR classification pipeline. It includes fixed versions of problematic functions, test data, and scripts to run the pipeline in different configurations.

## Key Files and Scripts

### Main Test Scripts
- `run_test.m` - Main test script that uses fixed versions of problematic functions
- `run_pipeline_test_simplified.m` - Simplified and consolidated test script
- `run_pipeline_test_debug.m` - Debug version with detailed output
- `run_test_with_debug.m` - Wrapper script with error handling for debugging

### Configuration
- `test_config.m` - Test-specific configuration with adjusted parameters

### Fixed Functions
- `load_and_prepare_data_fixed.m` - Fixed version of data loading function
- `perform_feature_selection_fixed.m` - Fixed version of feature selection function
- `run_cross_validation_complete.m` - Complete working implementation of cross-validation

### Utilities
- `extract_real_test_data.m` - Extracts subset of real data for testing
- `fix_categorical_issues.m` - Fixes issues with categorical variables
- `cleanup_test_directory.m` - Script to clean up and organize test files

## Test Data
A subset of real data is used for testing, with 14 training samples and 6 test samples:
- `data/data_table_train.mat`
- `data/data_table_test.mat`
- `data/wavenumbers.mat`

## Results and Outputs
Test outputs are stored in:
- `results/test_outputs/` - Consolidated test output files

## Running Tests
To run the test pipeline:
```matlab
% Simple test run
run_test

% Detailed debug output
run_test_with_debug

% Run simplified consolidated version
run_pipeline_test_simplified
```

## Notes on Fixed vs Simple Scripts
- **Fixed scripts** (`*_fixed.m`) contain bug fixes to address specific issues like categorical variable handling and PCA application.
- **Simple scripts** (`*_simple.m`) are simplified versions for faster testing and debugging.
- `run_cross_validation_complete.m` is the most comprehensive and stable cross-validation implementation.

## Last Updated
October 21, 2025