# Meningioma FTIR Classification Pipeline

Machine learning pipeline for classifying meningioma tumors (WHO Grade 1 vs 3) using FTIR spectroscopy data.

## 🚀 Quick Start

```matlab
% Navigate to project root
cd 'c:\Users\Franz\OneDrive\01_Promotion\01 Data\new-pipeline'

% Add source to path
addpath('src/meningioma_ftir_pipeline');

% Run validation tests
test_direct_pipeline

% Run full pipeline
run_pipeline_direct()
```

## 📖 Documentation

All documentation is in the **[`docs/`](docs/)** folder:

- **[docs/REFACTORED_PIPELINE.md](docs/REFACTORED_PIPELINE.md)** - Current pipeline usage guide
- **[docs/DEVELOPMENT_HISTORY.md](docs/DEVELOPMENT_HISTORY.md)** - Complete development history
- **[docs/README.md](docs/README.md)** - Full documentation index

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

- ✅ **Patient-wise stratified CV** - Prevents data leakage
- ✅ **Individual spectrum prediction** - No averaging before classification
- ✅ **Direct table access** - Fast, efficient data loading
- ✅ **Quality control** - Automated QC with multiple metrics
- ✅ **Multiple classifiers** - LDA, PLSDA, SVM, Random Forest

## 📊 Data

- **Training**: 44 probes from 37 patients (~32,470 spectra)
- **Test**: 32 probes from 23 patients (~24,115 spectra)
- **Classes**: WHO-1 (benign) vs WHO-3 (malignant)

## 🔬 Scientific References

- Baker et al. (2014). *Nature Protocols* 9(8):1771-1791
- Greener et al. (2022). *Nature Reviews Molecular Cell Biology* 23:40-55

## 📝 Version

**Current Version**: 3.0 (October 2025)
- Refactored for direct table access
- Eliminated intermediate files
- 57% code reduction
- 63% faster data loading

## 🆘 Support

For detailed information, see the [documentation](docs/README.md).

---

*Pipeline for FTIR-based meningioma classification research*
