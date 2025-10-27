# 📁 Project Directory Structure - Patient-Wise CV Implementation

## Overview
This document shows where all new files are located and how they relate to each other.

```
new-pipeline/
│
├── 📘 COMPLETE_OVERVIEW.md              ⭐ START HERE - Executive summary
├── 📘 QUICK_START.md                    ⭐ Quick start guide (5 min)
├── 📘 PATIENT_WISE_CV_README.md         📖 Main documentation (15 min)
├── 📘 IMPLEMENTATION_SUMMARY.md         🔧 Technical details (10 min)
│
├── data/                                 📊 Data files (user provided)
│   ├── data_table_train.mat
│   ├── data_table_test.mat
│   └── wavenumbers.mat
│
├── src/
│   └── meningioma_ftir_pipeline/
│       │
│       ├── 🔵 ORIGINAL PIPELINE FILES (unchanged):
│       ├── config.m                     ⚙️ Configuration (UPDATED with patient-wise options)
│       ├── load_and_prepare_data.m      ⚠️ OLD - averages spectra
│       ├── run_cross_validation.m       ⚠️ OLD - not patient-wise
│       ├── run_full_pipeline.m          ⚠️ OLD - uses averaging
│       ├── quality_control_analysis.m
│       ├── perform_feature_selection.m
│       ├── train_final_model.m
│       ├── evaluate_test_set.m
│       ├── generate_report.m
│       ├── feature_engineering.m
│       ├── train_model.m
│       └── helper_functions.m
│       │
│       ├── 🟢 NEW PATIENT-WISE FILES (core implementation):
│       ├── load_and_prepare_data_patientwise.m      ⭐ Data loading (NO averaging)
│       ├── patientwise_cv_functions.m               ⭐ CV fold creation & aggregation
│       ├── run_patientwise_cross_validation.m       ⭐ Main CV runner
│       ├── patientwise_metrics.m                    ⭐ Dual-level metrics
│       ├── patientwise_visualization.m              ⭐ Figures & Excel export
│       ├── run_full_pipeline_patientwise.m          ⭐ Complete pipeline wrapper
│       └── test_patientwise_implementation.m        🧪 Validation tests
│       │
│       └── utils/
│           └── log_message.m
│
├── models/
│   └── meningioma_ftir_pipeline/
│       └── pca_model.mat                📐 PCA transformation (generated)
│
├── results/
│   └── meningioma_ftir_pipeline/
│       ├── qc/
│       │   └── qc_flags.mat             ✓ Quality control results
│       │
│       ├── 🟢 NEW PATIENT-WISE OUTPUTS:
│       ├── patientwise_data.mat                     📊 Patient-indexed data
│       ├── cv_results_patientwise.mat               💾 Full CV results
│       ├── cv_results_patientwise.xlsx              📑 Patient predictions (Excel)
│       ├── cv_results_patientwise_summary.txt       📄 Summary statistics
│       └── patient_confidence_analysis.png/.fig     📊 6-panel visualization
│       │
│       └── 🔵 ORIGINAL OUTPUTS (from old pipeline):
│           ├── preprocessed_data.mat
│           └── X_train_pca.mat
│
└── archive/                              🗄️ Backup of development/testing files
    ├── main/
    └── test/
```

---

## 📚 Documentation Files (Read in this order)

### 1. Executive Summary
- **File**: `COMPLETE_OVERVIEW.md`
- **Purpose**: High-level overview of entire implementation
- **Read time**: 5-10 minutes
- **Audience**: Everyone
- **Key sections**: Quick start, deliverables, validation

### 2. Quick Start Guide
- **File**: `QUICK_START.md`
- **Purpose**: Get started in 3 steps
- **Read time**: 5 minutes
- **Audience**: New users
- **Key sections**: Installation, usage, troubleshooting

### 3. Main Documentation
- **File**: `PATIENT_WISE_CV_README.md`
- **Purpose**: Comprehensive user guide
- **Read time**: 15 minutes
- **Audience**: All users
- **Key sections**: Features, usage, data structures, metrics

### 4. Implementation Details
- **File**: `IMPLEMENTATION_SUMMARY.md`
- **Purpose**: Technical implementation details
- **Read time**: 10 minutes
- **Audience**: Developers, reviewers
- **Key sections**: What changed, code structure, validation

---

## 🔵 Core Implementation Files

### Data Loading
**`load_and_prepare_data_patientwise.m`** (283 lines)
```
Purpose: Load raw data into patient-indexed structure
Key function: load_and_prepare_data_patientwise(cfg)
Output: patientwise_data.mat
Key feature: NO AVERAGING - all spectra preserved
```

### CV Functions
**`patientwise_cv_functions.m`** (285 lines)
```
Functions:
  - createPatientWiseStratifiedCV()     → Create K-fold splits (patient-level)
  - extractSpectraForFold()             → Extract train/test spectra
  - aggregatePredictionsPerPatient()    → Majority voting
```

### Main CV Runner
**`run_patientwise_cross_validation.m`** (286 lines)
```
Purpose: Run complete cross-validation loop
Key function: run_patientwise_cross_validation(cfg)
Input: patientwise_data.mat
Output: cv_results_patientwise.mat
Steps: Preprocessing → Training → Prediction → Aggregation → Metrics
```

### Metrics Computation
**`patientwise_metrics.m`** (299 lines)
```
Functions:
  - computeMetrics()           → Dual-level metrics
  - displayFoldResults()       → Fold-wise display
  - aggregateCVResults()       → Cross-fold aggregation
Key feature: Patient-level metrics are PRIMARY
```

### Visualization & Export
**`patientwise_visualization.m`** (265 lines)
```
Functions:
  - visualizePatientConfidence()  → 6-panel figure
  - exportDetailedResults()       → Excel + text export
Output: patient_confidence_analysis.png, cv_results_patientwise.xlsx
```

### Pipeline Wrapper
**`run_full_pipeline_patientwise.m`** (202 lines)
```
Purpose: Complete end-to-end pipeline
Key function: run_full_pipeline_patientwise(perform_qc, perform_pca)
Calls: All above functions in sequence
Output: All result files
```

### Validation Tests
**`test_patientwise_implementation.m`** (140 lines)
```
Purpose: Automated validation of implementation
Key function: test_patientwise_implementation()
Tests: 6 validation checks
Use: Run before first use to verify setup
```

---

## 📊 Data Flow Diagram

```
RAW DATA
├── data_table_train.mat (44 patients × ~768 spectra each)
├── data_table_test.mat
└── wavenumbers.mat
         ↓
    [load_and_prepare_data_patientwise.m]
         ↓
PATIENT-WISE DATA
└── patientwise_data.mat
    ├── trainingData.patientData(i).spectra [768 × 441]
    └── testData.patientData(i).spectra [768 × 441]
         ↓
    [createPatientWiseStratifiedCV]
         ↓
CV FOLDS (Patient-level)
└── cvFolds(k).trainPatientIdx
└── cvFolds(k).testPatientIdx
         ↓
    [extractSpectraForFold]
         ↓
TRAIN/TEST SPECTRA
├── X_train: [~27k × 441] (all spectra from train patients)
└── X_test: [~6.7k × 441] (all spectra from test patients)
         ↓
    [Preprocessing + PCA]
         ↓
    [Train Classifier]
         ↓
INDIVIDUAL PREDICTIONS
└── y_pred_spectra: [~6.7k × 1] (one per spectrum)
         ↓
    [aggregatePredictionsPerPatient]
         ↓
PATIENT-LEVEL PREDICTIONS
└── patientPredictions (majority vote per patient)
         ↓
    [computeMetrics]
         ↓
RESULTS
├── cv_results_patientwise.mat
├── cv_results_patientwise.xlsx
├── cv_results_patientwise_summary.txt
└── patient_confidence_analysis.png
```

---

## 🎯 File Usage Guide

### I want to... → Use this file:

| Task | File to Use | Command |
|------|-------------|---------|
| **Get started quickly** | `QUICK_START.md` | Read documentation |
| **Understand implementation** | `PATIENT_WISE_CV_README.md` | Read documentation |
| **Run full pipeline** | `run_full_pipeline_patientwise.m` | `run_full_pipeline_patientwise()` |
| **Test setup** | `test_patientwise_implementation.m` | `test_patientwise_implementation()` |
| **Load patient data only** | `load_and_prepare_data_patientwise.m` | `load_and_prepare_data_patientwise(cfg)` |
| **Run CV only** | `run_patientwise_cross_validation.m` | `run_patientwise_cross_validation(cfg)` |
| **Create visualizations** | `patientwise_visualization.m` | `visualizePatientConfidence(cvResults, dir)` |
| **Export to Excel** | `patientwise_visualization.m` | `exportDetailedResults(cvResults, ...)` |
| **Configure settings** | `config.m` | Edit file, then `cfg = config()` |

---

## 🔄 Relationship to Original Pipeline

### Files Kept Unchanged
- `quality_control_analysis.m` - Still used for QC
- `feature_engineering.m` - Helper functions
- `helper_functions.m` - Utility functions

### Files Replaced (Functionality)
| Original | New Patient-Wise | Change |
|----------|------------------|--------|
| `load_and_prepare_data.m` | `load_and_prepare_data_patientwise.m` | NO averaging |
| `run_cross_validation.m` | `run_patientwise_cross_validation.m` | Patient-level CV |
| `run_full_pipeline.m` | `run_full_pipeline_patientwise.m` | Patient-wise wrapper |

### Files Not Yet Adapted
- `train_final_model.m` - Can be used after CV
- `evaluate_test_set.m` - Can be adapted for patient-wise test set
- `generate_report.m` - Can be adapted for patient-wise results

---

## 📦 Output Files Explained

### Generated by Pipeline

**`patientwise_data.mat`** (Intermediate)
- Patient-indexed data structure
- All spectra preserved (no averaging)
- Used by CV functions

**`cv_results_patientwise.mat`** (Main results)
- Complete MATLAB structure
- All CV folds
- Trained models
- Predictions
- Metrics

**`cv_results_patientwise.xlsx`** ⭐ (Clinical review)
- One row per patient
- Predictions, confidence, interpretation flags
- **Most important for clinical use**

**`cv_results_patientwise_summary.txt`** (Summary)
- Performance metrics (mean ± SD, 95% CI)
- Confidence distribution
- Clinical interpretation counts

**`patient_confidence_analysis.png`** (Visualization)
- 6-panel figure
- Confidence, entropy, agreement, confusion matrix

---

## 🧪 Testing & Validation

### Validation Test File
**`test_patientwise_implementation.m`**

**6 Tests:**
1. Configuration validity
2. Data file existence  
3. Raw data structure
4. Patient-wise data loading
5. CV fold creation (no overlap)
6. Spectrum extraction

**Usage:**
```matlab
test_patientwise_implementation();
% All tests should show ✓ PASS
```

---

## ⚙️ Configuration File

**`config.m`** (Updated)

**New Settings for Patient-Wise CV:**
```matlab
cfg.classifiers.primary_type = 'SVM';  % Classifier to use
cfg.classifiers.svm_C = 1;             % SVM hyperparameter
cfg.classifiers.svm_gamma = 'auto';    % SVM hyperparameter
cfg.classifiers.rf_n_trees = 100;      % RandomForest trees
```

**Unchanged Settings:**
```matlab
cfg.cv.n_folds = 5;                    % Number of CV folds
cfg.random_seed = 42;                  % Reproducibility
cfg.pca.variance_threshold = 0.95;     % PCA variance to keep
```

---

## 🗺️ Navigation Tips

### Starting Point
1. Read `COMPLETE_OVERVIEW.md` (this file is the executive summary version)
2. Follow `QUICK_START.md` for hands-on
3. Refer to `PATIENT_WISE_CV_README.md` for details

### For Development
1. Check `IMPLEMENTATION_SUMMARY.md` for technical details
2. Read function comments in source files
3. Use `test_patientwise_implementation.m` to validate changes

### For Clinical Use
1. Run `run_full_pipeline_patientwise()`
2. Open `cv_results_patientwise.xlsx` in Excel
3. Filter by `InterpretationFlag` column
4. Review high-confidence incorrect cases

---

## 📞 Quick Reference

### Most Important Files

**For Users:**
1. `QUICK_START.md` - How to use
2. `run_full_pipeline_patientwise.m` - What to run
3. `cv_results_patientwise.xlsx` - Results to review

**For Developers:**
1. `IMPLEMENTATION_SUMMARY.md` - What was implemented
2. `patientwise_cv_functions.m` - Core CV logic
3. `test_patientwise_implementation.m` - Validation

**For Documentation:**
1. `PATIENT_WISE_CV_README.md` - Main docs
2. Code comments in all `.m` files
3. `.github/copilot-instructions.md` - Pipeline context

---

## 🎓 Learning Path

### Beginner (Just want to use it)
1. `QUICK_START.md` (5 min)
2. `run_full_pipeline_patientwise()` command
3. Review Excel output

### Intermediate (Want to understand it)
1. `COMPLETE_OVERVIEW.md` (10 min)
2. `PATIENT_WISE_CV_README.md` (15 min)
3. Run with custom settings

### Advanced (Want to modify it)
1. `IMPLEMENTATION_SUMMARY.md` (10 min)
2. Read function source code
3. Run validation tests
4. Make modifications

---

**Directory Structure Document**: Complete ✅

This file helps you navigate all new files and understand how they work together!
