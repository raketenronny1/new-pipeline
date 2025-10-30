# Meningioma FTIR Classification Pipeline

**Production-ready machine learning pipeline for classifying meningioma tumors (WHO Grade 1 vs 3) using FTIR spectroscopy data.**

**Version**: 5.0 (October 2025) - Modular Architecture with Comprehensive Testing  
**Status**: Production-ready with object-oriented design and 100% test coverage

---

## � Overview

This repository contains a complete MATLAB-based pipeline for analyzing FTIR (Fourier Transform Infrared) spectroscopy data from meningioma tumor samples. The pipeline implements patient-level stratified cross-validation with multiple machine learning classifiers to distinguish between WHO Grade 1 (benign) and Grade 3 (malignant) meningiomas.

### Key Capabilities
- **Data Loading**: Flexible data loader with auto-detection of field names and formats
- **Preprocessing**: Modular FTIR spectral preprocessing (binning, smoothing, normalization, derivatives)
- **Classification**: Multiple classifiers with hyperparameter optimization (PCA-LDA, SVM-RBF, PLS-DA, Random Forest)
- **Validation**: Patient-stratified cross-validation preventing data leakage
- **Metrics**: Comprehensive performance metrics at both spectrum and patient levels
- **Reporting**: Automated report generation with visualizations and statistical analysis

---

## 🚀 Quick Start

### Prerequisites
- MATLAB R2019b or later
- Statistics and Machine Learning Toolbox
- Parallel Computing Toolbox (optional, for parallel execution)

### Basic Usage

```matlab
% Navigate to project root
cd 'c:\Users\Franz\OneDrive\01_Promotion\01 Data\new-pipeline'

% Run the complete pipeline
run_full_pipeline
```

The pipeline performs:
1. ✅ Data verification and loading
2. ✅ Spectral preprocessing (BSNCX configurations)
3. ✅ Patient-stratified cross-validation
4. ✅ Model training and evaluation
5. ✅ Results export and visualization

### Data Preparation (First-Time Setup)

```matlab
% Clean data files to contain only raw spectra
% Run this ONCE before first pipeline execution
prepare_data
```

This script:
- Removes preprocessed columns from data files
- Calculates mean raw spectra per sample
- Validates train/test split integrity
- Creates backups of original files

---

## 📁 Project Structure

```
new-pipeline/
├── data/                           # Raw FTIR data files
│   ├── data_table_train.mat       # Training data (52 samples, 42 patients)
│   ├── data_table_test.mat        # Test data (24 samples, 15 patients)
│   ├── dataset_complete.mat       # Complete dataset (115 samples)
│   ├── split_info.mat             # Train/test split information
│   └── wavenumbers.mat            # FTIR wavenumber axis
│
├── src/                            # Source code (modular architecture)
│   ├── classifiers/               # Classifier implementations
│   │   └── ClassifierWrapper.m    # Unified interface for all classifiers
│   ├── metrics/                   # Performance metrics
│   │   └── MetricsCalculator.m    # Spectrum and patient-level metrics
│   ├── preprocessing/             # Spectral preprocessing
│   │   ├── PreprocessingPipeline.m    # Main preprocessing class
│   │   ├── apply_binning.m            # Wavenumber binning
│   │   ├── apply_sg_smoothing.m       # Savitzky-Golay smoothing
│   │   ├── apply_sg_derivative.m      # Derivative calculations
│   │   └── apply_vector_normalization.m # Vector normalization
│   ├── reporting/                 # Results and visualization
│   │   ├── ResultsAggregator.m    # Cross-validation results aggregation
│   │   ├── VisualizationTools.m   # Publication-quality plots
│   │   └── ReportGenerator.m      # Automated report generation
│   ├── utils/                     # Utility functions
│   │   ├── Config.m               # Configuration management
│   │   ├── DataLoader.m           # Data loading and validation
│   │   └── export_cv_results.m    # Results export utilities
│   └── validation/                # Cross-validation
│       └── CrossValidationEngine.m # Patient-stratified CV engine
│
├── tests/                          # Comprehensive test suite (63 unit tests)
│   ├── run_all_tests.m            # Master test runner
│   ├── test_config.m              # Config tests (9 tests)
│   ├── test_data_loader.m         # DataLoader tests (8 tests)
│   ├── test_preprocessing_pipeline.m # Preprocessing tests (12 tests)
│   ├── test_classifier_wrapper.m  # Classifier tests (12 tests)
│   ├── test_cross_validation_engine.m # CV tests (8 tests)
│   ├── test_metrics_calculator.m  # Metrics tests (8 tests)
│   ├── test_results_aggregator.m  # Aggregator tests (6 tests)
│   └── test_integration_mini.m    # End-to-end integration test
│
├── models/                         # Trained models (generated)
├── results/                        # Analysis results (generated)
│
├── archive/                        # Historical versions
│   ├── old_pipeline_2025-10-29/   # Previous version 4.0
│   ├── main/                      # Main branch archives
│   └── test/                      # Testing archives
│
├── docs/                           # Documentation
│   └── GIT_LFS_SETUP.md           # Git LFS configuration guide
│
├── run_full_pipeline.m            # Main pipeline script
├── prepare_data.m                 # Data preparation script
├── README.md                      # This file
├── IMPLEMENTATION_SUMMARY.md      # Technical implementation details
└── CLEANUP_SUMMARY.md             # Code cleanup history
```

---

## 🏗️ Architecture

### Component Hierarchy

```
Config (Singleton Configuration)
    ↓
DataLoader → PreprocessingPipeline → ClassifierWrapper
                                           ↓
                                CrossValidationEngine
                                           ↓
                                  MetricsCalculator
                                           ↓
                                  ResultsAggregator
                                           ↓
                          VisualizationTools + ReportGenerator
```

### Core Components

#### 1. **Config** (`src/utils/Config.m`)
- Singleton pattern for centralized configuration
- Manages all pipeline parameters
- Supports struct-based or default initialization

#### 2. **DataLoader** (`src/utils/DataLoader.m`)
- Auto-detects data table field names
- Handles multiple spectra per sample
- Validates patient-level data integrity
- Prevents train/test patient overlap

#### 3. **PreprocessingPipeline** (`src/preprocessing/PreprocessingPipeline.m`)
- BSNCX notation for preprocessing configurations:
  - **B**inning (0=none, 1=bin)
  - **S**moothing (0=none, 1=Savitzky-Golay)
  - **N**ormalization (2=vector normalize)
  - **C**orrection (0=none, 1=1st deriv, 2=2nd deriv)
  - **X** (placeholder)
- Fit/transform separation prevents data leakage

#### 4. **ClassifierWrapper** (`src/classifiers/ClassifierWrapper.m`)
Unified interface for four classifiers:
- **PCA-LDA**: Principal Component Analysis + Linear Discriminant Analysis
- **SVM-RBF**: Support Vector Machine with RBF kernel
- **PLS-DA**: Partial Least Squares Discriminant Analysis
- **Random Forest**: Ensemble of decision trees

#### 5. **CrossValidationEngine** (`src/validation/CrossValidationEngine.m`)
- Patient-level stratified cross-validation
- Nested loop: Permutations → Classifiers → Repeats → Folds
- Zero patient overlap enforcement
- Parallel execution support

#### 6. **MetricsCalculator** (`src/metrics/MetricsCalculator.m`)
Computes metrics at two levels:
- **Spectrum-level**: Accuracy, sensitivity, specificity, F1-score, AUC-ROC
- **Patient-level**: Majority vote aggregation across samples

#### 7. **ResultsAggregator** (`src/reporting/ResultsAggregator.m`)
- Aggregates results across folds and repeats
- Computes mean/std/median statistics
- Identifies best configurations
- Exports to CSV and MATLAB formats

#### 8. **VisualizationTools** (`src/reporting/VisualizationTools.m`)
- Confusion matrices
- ROC curves
- Performance heatmaps
- Classifier comparison plots
- Publication-quality output (PNG/PDF/EPS/FIG)

#### 9. **ReportGenerator** (`src/reporting/ReportGenerator.m`)
- Orchestrates full analysis pipeline
- Generates comprehensive reports
- Creates visualizations
- Exports statistical summaries

---

## ✨ Key Features

### Data Leakage Prevention
- ✅ **Patient-level CV splitting** (not sample-level)
- ✅ **Preprocessing fit/transform separation**
- ✅ **Parameter freezing for test set**
- ✅ **Patient overlap detection with error throwing**

### MATLAB API Compliance
- ✅ All classifiers use official MATLAB functions
- ✅ Verified against MATLAB documentation
- ✅ Proper parameter naming conventions

### Reproducibility
- ✅ Random seed control for all stochastic operations
- ✅ Complete provenance tracking
- ✅ Version-controlled configuration
- ✅ Automated testing suite

### Comprehensive Testing
- ✅ **63 unit tests** across 7 components (100% passing)
- ✅ **End-to-end integration test** with synthetic data
- ✅ Automated test runner (`tests/run_all_tests.m`)

---

## 📊 Dataset Information

- **Training Set**: 52 samples from 42 patients (~32,470 individual spectra)
- **Test Set**: 24 samples from 15 patients (~24,115 individual spectra)
- **Total**: 76 samples from 57 patients
- **Classes**: 
  - WHO Grade 1 (benign meningioma)
  - WHO Grade 3 (malignant/anaplastic meningioma)
- **Features**: 110 wavenumbers (FTIR spectral bins)
- **Data Format**: MATLAB tables with cell arrays of spectra

---

## 🧪 Testing

### Run All Tests

```matlab
cd tests
run_all_tests
```

### Test Coverage Summary

| Component | Tests | Status |
|-----------|-------|--------|
| Config | 9 | ✅ ALL PASS |
| DataLoader | 8 | ✅ ALL PASS |
| PreprocessingPipeline | 12 | ✅ ALL PASS |
| ClassifierWrapper | 12 | ✅ ALL PASS |
| CrossValidationEngine | 8 | ✅ ALL PASS |
| MetricsCalculator | 8 | ✅ ALL PASS |
| ResultsAggregator | 6 | ✅ ALL PASS |
| **TOTAL** | **63** | **✅ 100%** |

### Integration Test
- End-to-end pipeline with synthetic data (36 samples, 12 patients)
- 4 configurations (2 permutations × 2 classifiers)
- Full report generation with visualizations

---

## 📈 Results and Outputs

Pipeline execution generates:

```
results/
├── plots/
│   ├── confusion_matrix_*.png
│   ├── roc_curve_*.png
│   ├── performance_heatmap_*.png
│   └── classifier_comparison_*.png
├── spectrum_level_summary.mat
├── spectrum_level_results.csv
├── patient_level_summary.mat
├── patient_level_results.csv
├── best_configurations.mat
└── analysis_report.txt
```

---

## 🔬 Scientific Background

This pipeline implements best practices for FTIR spectroscopy analysis in biomedical applications:

- **Preprocessing**: Based on Baker et al. (2014) guidelines for IR spectroscopy
- **Machine Learning**: Cross-validation practices from Greener et al. (2022)
- **Clinical Application**: WHO grading system for meningioma classification

### References
- Baker, M. J., et al. (2014). *Using Fourier transform IR spectroscopy to analyze biological materials.* Nature Protocols, 9(8), 1771-1791.
- Greener, J. G., et al. (2022). *A guide to machine learning for biologists.* Nature Reviews Molecular Cell Biology, 23, 40-55.

---

## 📝 Version History

### **Version 5.0** (October 30, 2025)
- 🏗️ **Modular architecture** - Object-oriented design with 9 core classes
- 🧪 **Comprehensive testing** - 63 unit tests + integration tests (100% passing)
- 🔒 **Data leakage prevention** - Patient-level CV with preprocessing isolation
- 📊 **Enhanced reporting** - Publication-quality visualizations
- 📦 **Better organization** - Clear src/ structure with logical grouping
- 🔧 **Data preparation** - `prepare_data.m` script for proper data cleaning

### **Version 4.0** (October 24, 2025)
- 🎯 **Unified pipeline** - Single entry point
- 🔬 **EDA integration** - Outlier detection via T²-Q statistics
- 🧹 **Code consolidation** - Streamlined components

---

## 🛠️ Development

### Adding New Classifiers

Extend `ClassifierWrapper` with new classifier types:

```matlab
% In ClassifierWrapper.m, add to trainModel method:
case 'new_classifier'
    model = your_training_function(X, y, params);
    % Ensure predict returns scores compatible with existing interface
```

### Adding New Preprocessing Steps

Add functions to `src/preprocessing/` following the pattern:

```matlab
function [X_out, params] = apply_new_preprocessing(X_in, params)
    % Your preprocessing logic
    X_out = process(X_in);
    params.new_param = value;  % Store learned parameters
end
```

### Running Tests During Development

```matlab
% Run specific test file
test_classifier_wrapper

% Run all tests
cd tests; run_all_tests
```

---

## 📧 Support and Contributing

For questions, issues, or contributions, please refer to:
- **Implementation Details**: `IMPLEMENTATION_SUMMARY.md`
- **Code Cleanup History**: `CLEANUP_SUMMARY.md`
- **AI Assistant Guidelines**: `.github/copilot-instructions.md`

---

## 📄 License

See `src/utils/license.txt` for licensing information.
- 📚 **Streamlined docs** - Clear, non-redundant documentation

**What Changed**:
- Replaced `run_pipeline_direct()`, `run_pipeline_with_eda()`, `run_full_eda()` → `run_pipeline()`
- Replaced `load_data_direct()`, `load_data_with_eda()`, `load_and_prepare_data()` → `load_pipeline_data()`
- Moved test files from root to `tests/` directory
- Extracted common export functions to `src/utils/`
- All old versions backed up to `archive/2025-10-24_refactoring/`

## 🆘 Support

For detailed information, see the [documentation](docs/README.md).

---

*Pipeline for FTIR-based meningioma classification research*
