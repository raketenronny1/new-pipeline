# Meningioma FTIR Classification Pipeline

**Unified machine learning pipeline for classifying meningioma tumors (WHO Grade 1 vs 3) using FTIR spectroscopy data.**

**Version**: 4.0 (October 2025) - Refactored & Streamlined  
**Status**: Production-ready with EDA-based outlier detection

---

## 🚀 Quick Start

> **⚠️ Important**: This repository uses **Git LFS** for large data files. See [Git LFS Setup Guide](docs/GIT_LFS_SETUP.md) if you're cloning for the first time.

### **Option 1: Run Complete Pipeline (Recommended)**

```matlab
% Navigate to project root
cd 'c:\Users\Franz\OneDrive\01_Promotion\01 Data\new-pipeline'

% Add source to path
addpath('src/meningioma_ftir_pipeline');

% Run complete pipeline (EDA + CV + Export)
run_pipeline()
```

That's it! The pipeline will:
1. Run exploratory data analysis (EDA) with outlier detection
2. Load and filter data based on EDA results
3. Perform patient-stratified cross-validation
4. Export results to Excel and text files

### **Option 2: Step-by-Step Execution**

```matlab
% Step 1: Run EDA (only needed once)
run_eda()

% Step 2: Run pipeline without repeating EDA
run_pipeline('RunEDA', false)

% Quick test (3 folds, 10 repeats)
run_pipeline('RunEDA', false, 'NFolds', 3, 'NRepeats', 10)
```

---

## 📖 Documentation

**Essential Docs** (Read these first):
- **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)** - Complete usage guide (NEW)
- **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)** - Function reference (NEW)

**Additional Resources**:
- **[docs/DEVELOPMENT_HISTORY.md](docs/DEVELOPMENT_HISTORY.md)** - Development history & design decisions
- **[docs/GIT_LFS_SETUP.md](docs/GIT_LFS_SETUP.md)** - Git LFS configuration guide (NEW)
- **[.github/copilot-instructions.md](.github/copilot-instructions.md)** - Pipeline architecture overview

## 📁 Project Structure

```
new-pipeline/
├── data/                          # Raw data (MATLAB .mat files)
├── models/                        # Trained models
├── results/                       # Analysis results
├── src/
│   └── meningioma_ftir_pipeline/  # Source code
├── docs/                          # 📚 Documentation
├── archive/                       # Historical code versions
└── README.md                      # This file
```

## ✨ Key Features

- ✅ **EDA-based outlier detection** - T² and Q statistics for robust quality control
- ✅ **Patient-wise stratified CV** - Prevents data leakage across patients
- ✅ **Individual spectrum prediction** - No averaging, preserves intra-sample variability
- ✅ **Unified interface** - Single entry point with flexible options
- ✅ **Multiple classifiers** - LDA (with PCA), PLS-DA, SVM-RBF, Random Forest
- ✅ **Hyperparameter optimization** - Bayesian optimization for all classifiers
- ✅ **Comprehensive exports** - Excel, text summaries, MATLAB structures

## 📊 Data

- **Training**: 44 probes from 37 patients (~32,470 spectra)
- **Test**: 32 probes from 23 patients (~24,115 spectra)
- **Classes**: WHO-1 (benign) vs WHO-3 (malignant)

## 🔬 Scientific References

- Baker et al. (2014). *Nature Protocols* 9(8):1771-1791
- Greener et al. (2022). *Nature Reviews Molecular Cell Biology* 23:40-55

## 📝 Recent Updates

**Version 4.0** (October 24, 2025)
- 🎯 **Unified pipeline** - Single `run_pipeline()` entry point
- 🔬 **EDA integration** - Outlier detection via T²-Q statistics
- 🧹 **Code consolidation** - 3 data loaders → 1, multiple runners → 1
- 📦 **Better organization** - Tests in `tests/`, utilities in `src/utils/`
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
