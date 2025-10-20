# Meningioma FT-IR Classification Pipeline

This pipeline implements a machine learning workflow for classifying meningioma samples based on FT-IR spectroscopy data. The pipeline includes comprehensive quality control, feature selection, cross-validation, and model evaluation.

## Project Structure

```
project_root/
├── data/                      # Input data files
│   ├── data_table_train.mat   # Training data table
│   ├── data_table_test.mat    # Test data table
│   └── wavenumbers.mat        # Wavenumber vector
├── models/
│   └── meningioma_ftir_pipeline/  # Trained models and parameters
├── results/
│   └── meningioma_ftir_pipeline/  # Results and visualizations
│       └── qc/                    # Quality control outputs
└── src/
    └── meningioma_ftir_pipeline/  # Source code
```

## Dependencies

- MATLAB R2025b
- Required toolboxes:
  - Statistics and Machine Learning Toolbox
  - Signal Processing Toolbox

## Installation

1. Clone or download this repository
2. Place your data files in the `data/` directory
3. Ensure all required MATLAB toolboxes are installed

## Usage

1. Add the source directory to MATLAB path:
```matlab
addpath('src/meningioma_ftir_pipeline');
```

2. Run the complete pipeline:
```matlab
run_full_pipeline();
```

Or run individual phases:
```matlab
% Phase 0: Quality Control
quality_control_analysis();

% Phase 1: Data Loading
load_and_prepare_data();

% Phase 2: Feature Selection
perform_feature_selection();

% Phase 3: Cross-Validation
run_cross_validation();

% Phase 4: Final Model Training
train_final_model();

% Phase 5: Test Evaluation
evaluate_test_set();

% Phase 6: Report Generation
generate_report();
```

## Configuration

All pipeline parameters are configured in `config_meningioma_ftir_pipeline.m`. Key parameters include:

- Quality control thresholds
- PCA parameters
- Cross-validation settings
- Classifier hyperparameter grids

## Outputs

The pipeline generates:

1. Quality Control
   - QC report and metrics
   - QC visualizations
   - Cleaned data files

2. Model Development
   - PCA model and transformed data
   - Cross-validation results
   - Final trained model

3. Evaluation
   - Test set performance metrics
   - ROC curves and confusion matrices
   - Performance visualizations

4. Documentation
   - Comprehensive final report
   - Methods section for publication
   - Complete execution log

## Validation

The pipeline implements strict validation practices:

- Complete train/test separation
- No data leakage
- Transparent reporting of all results
- Reproducible random seeds

## Contributing

Please ensure any contributions:
1. Maintain strict train/test separation
2. Document all changes
3. Follow existing code style
4. Include appropriate error handling

## License

[Add your license information here]

## Contact

[Add your contact information here]