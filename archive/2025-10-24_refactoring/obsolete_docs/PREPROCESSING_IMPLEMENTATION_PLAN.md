# FTIR Data Preprocessing Implementation Plan

**Date:** October 24, 2025  
**Version:** 1.0  
**Status:** Implementation Ready

---

## Overview

This document outlines the implementation plan for dual preprocessing approaches in the FTIR spectroscopy meningioma classification pipeline. The goal is to recreate and enhance the `dataset_men` table with two distinct preprocessing strategies while maintaining compatibility with existing pipeline components.

---

## Data Sources

### Input Files
- **`allspekTable.mat`**: 351 position-level measurements
  - Each row: one position measurement (Pos1, Pos2, or Pos3)
  - `RawSpectrum`: 256×441 double matrix per position
  - `Proben_ID_str`: Sample identifier
  - `Position`: Position label
  
- **`metadata_all_patients.mat`**: 123 patient records
  - Patient demographics and clinical data
  - Methylation classification data
  - Links to allspekTable via `Proben_ID`

- **`wavenumbers.mat`**: Wavenumber vector
  - Fingerprint region: 1800-950 cm⁻¹
  - 441 data points

### Output Files
- **`dataset_complete.mat`**: Complete probe-level dataset (`dataset_men`)
- **`data_table_train.mat`**: Training subset
- **`data_table_test.mat`**: Test subset

---

## Preprocessing Approaches

### Background: Already Applied in OPUS Software
The following preprocessing steps have **already been applied** to the data in `allspekTable.mat`:
1. ✓ CO₂ correction
2. ✓ Water vapor correction  
3. ✓ 4×4 pixel binning (hardware-level)
4. ✓ Spectral window selection (1800-950 cm⁻¹)
5. ✓ Rubberband baseline correction

### To Be Implemented in MATLAB

#### **Approach 1 (PP1) - Standard Pipeline** ⭐ DEFAULT
**Purpose:** Minimal processing, maximum information retention

**Processing Steps:**
1. **No binning** - Use full spectral resolution
2. **No smoothing** - Preserve spectral features
3. **Vector Normalization (L2 norm = 1)**
   - Normalize each spectrum vector to unit length
   - Formula: `spectrum_norm = spectrum / norm(spectrum, 2)`
   - Effect: Removes intensity scaling, focuses on spectral shape
4. **Second Derivative Baseline Correction**
   - Method: Savitzky-Golay filter
   - Window size: 5 points
   - Polynomial order: 2
   - Derivative order: 2
   - Effect: Removes baseline variations, enhances peak resolution

**Storage Column:** `CombinedSpectra_PP1`  
**Default for Analysis:** ✓ Yes

---

#### **Approach 2 (PP2) - Enhanced Pipeline**
**Purpose:** Noise reduction through binning and smoothing

**Processing Steps:**
1. **Binning (Factor 4)**
   - Average neighboring wavenumber points in groups of 4
   - Reduces spectral resolution by 4×
   - Effect: Improves SNR, reduces computational load
   - New spectrum length: 441 → ~110 points

2. **Savitzky-Golay Smoothing**
   - Window size: 11 points
   - Polynomial order: 2
   - **Derivative order: 0** (smoothing only, NOT derivative)
   - Effect: Reduces high-frequency noise while preserving peak shapes

3. **Vector Normalization (L2 norm = 1)**
   - Same as PP1
   - Formula: `spectrum_norm = spectrum / norm(spectrum, 2)`

4. **Second Derivative Baseline Correction**
   - Method: Savitzky-Golay filter
   - Window size: 5 points
   - Polynomial order: 2
   - Derivative order: 2
   - Effect: Removes baseline variations, enhances peak resolution

**Storage Column:** `CombinedSpectra_PP2`  
**Default for Analysis:** ✗ No (experimental/comparative)

---

## Implementation Architecture

### Directory Structure
```
src/
└── preprocessing/
    ├── apply_binning.m                    # Binning (factor N)
    ├── apply_vector_normalization.m       # L2 normalization
    ├── apply_sg_smoothing.m               # SG smoothing (derivative=0)
    ├── apply_sg_derivative.m              # SG second derivative
    ├── preprocess_spectra.m               # Main orchestrator
    └── create_preprocessing_config.m      # Config generator
```

### Main Pipeline Scripts
```
src/meningioma_ftir_pipeline/
├── prepare_ftir_dataset.m         # Main data preparation (replaces example code)
├── split_train_test.m             # Train/test splitting
└── visualize_preprocessing.m      # Visualization suite
```

---

## Dataset Schema: `dataset_men`

### Complete Variable List

| Variable | Type | Description |
|----------|------|-------------|
| `Diss_ID` | cellstr | Unique sample identifier (e.g., "MEN-001-01") |
| `Patient_ID` | string | Patient identifier (e.g., "MEN-001") |
| `Fall_ID` | double | Case/Fall number |
| `WHO_Grade` | categorical | WHO classification: WHO-1, WHO-2, WHO-3 |
| `Sex` | categorical | Male, Female |
| `Age` | double | Patient age in years |
| `Subtyp` | categorical | Histological subtype (fibro, meningo, trans, etc.) |
| `methylation_class` | categorical | DNA methylation class (if available) |
| `methylation_cluster` | categorical | DNA methylation cluster (if available) |
| `NumPositions` | double | Number of measured positions (typically 3) |
| `PositionSpectra` | cell | Cell array with position-level spectra details |
| `NumTotalSpectra` | double | Total number of spectra across all positions |
| **`CombinedRawSpectra`** | **cell** | **All raw unprocessed spectra (NEW)** |
| **`CombinedSpectra_PP1`** | **cell** | **Preprocessing Approach 1 (NEW, DEFAULT)** |
| **`CombinedSpectra_PP2`** | **cell** | **Preprocessing Approach 2 (NEW)** |
| **`MeanSpectrum_PP1`** | **cell** | **Mean representative spectrum (PP1)** |
| **`MeanSpectrum_PP2`** | **cell** | **Mean representative spectrum (PP2)** |

### Data Type Details
- **Categorical variables** use protected categories to prevent invalid assignments
- **Cell arrays** contain matrices where rows = individual spectra, columns = wavenumbers
- **CombinedRawSpectra**: Full resolution (441 wavenumbers)
- **CombinedSpectra_PP1**: Full resolution (441 wavenumbers)
- **CombinedSpectra_PP2**: Reduced resolution (~110 wavenumbers due to binning)

---

## Quality Control Framework

### 1. Spectrum-Level QC (Future Enhancement)
**Purpose:** Filter poor-quality individual spectra before aggregation

**Planned Metrics:**
- Signal-to-noise ratio (SNR)
- Baseline drift assessment
- Absorbance range validation
- Peak presence verification

**Implementation:** Similar to existing `quality_control_analysis.m` but applied at raw spectrum level

### 2. Representative Spectrum Creation
**Current Implementation:** Mean across all spectra per sample

**Methods Available:**
- Mean (current default)
- Median (robust to outliers)
- Trimmed mean (exclude extreme values)

**Storage:** `MeanSpectrum_PP1` and `MeanSpectrum_PP2`

### 3. Sample-Level QC (Future Enhancement)
**Purpose:** Identify outlier samples based on representative spectra

**Planned Metrics:**
- Mahalanobis distance in PC space
- Hotelling's T² statistic
- Leverage and studentized residuals

**Implementation:** Applied after PCA transformation

---

## Train/Test Split Strategy

### Decision: Maintain Separate Tables ✓

**Rationale:**
- Prevents accidental test set contamination
- Matches existing pipeline architecture
- Minimal refactoring required
- Clear separation of concerns
- Standard practice in ML workflows

### Split Rules
1. **Methylation samples** → Forced to TEST set (external validation)
2. **Balanced WHO-1/WHO-3** → Training set (equal class distribution)
3. **Remaining WHO-1/WHO-3** → Test set
4. **WHO-2 samples** → Excluded from train/test (future project)
5. **No patient overlap** between train and test

### Validation Checks
- Total sample count matches original
- Diss_ID sets are identical
- WHO-2 samples present in dataset but not in splits
- Class balance maintained in training set

---

## Visualization Suite

### 1. Before/After Preprocessing Comparison
**Script:** `visualize_preprocessing_comparison.m`

**Layout:** 3×1 tiled layout per sample
- Top panel: Raw spectrum
- Middle panel: PP1 preprocessed
- Bottom panel: PP2 preprocessed

**Samples Shown:**
- One per WHO grade (1, 2, 3)
- One per major subtype (meningo, fibro, trans, atyp, anap)
- Representative methylation sample

### 2. Step-by-Step Preprocessing Effects
**Script:** `visualize_preprocessing_steps.m`

**Layout:** Sequential transformation display
- Raw spectrum (baseline)
- After binning (PP2 only)
- After smoothing (PP2 only)
- After vector normalization
- After 2nd derivative

**Purpose:** Understand contribution of each processing step

### 3. PCA Visualization Framework
**Script:** `visualize_pca_framework.m`

**Plots:**
- PC1 vs PC2 scatter (color by WHO_Grade)
- PC1 vs PC3 scatter
- 3D plot: PC1-PC2-PC3
- Scree plot (variance explained)
- Loadings plot (top PCs)

**Flexibility:** Framework for future enhancements (e.g., interactive plots, multiple colorings)

---

## Implementation Work Packages

### **WP1: Preprocessing Functions** (`src/preprocessing/`)
**Duration:** High priority  
**Deliverables:**
- `apply_binning.m` - Spectral binning with configurable factor
- `apply_vector_normalization.m` - L2 normalization
- `apply_sg_smoothing.m` - SG smoothing (derivative order = 0)
- `apply_sg_derivative.m` - SG 2nd derivative
- `preprocess_spectra.m` - Main function with config-based routing
- `create_preprocessing_config.m` - Configuration structure generator

**Testing:** Unit tests with synthetic spectra

---

### **WP2: Data Preparation Pipeline**
**Duration:** High priority  
**Deliverables:**
- `prepare_ftir_dataset.m` - Modernized version of example code
- Dual preprocessing integration
- CombinedRawSpectra storage
- Position-level to probe-level aggregation
- Categorical variable formatting

**Testing:** Compare output with original `dataset_complete.mat`

---

### **WP3: Train/Test Split**
**Duration:** Medium priority  
**Deliverables:**
- `split_train_test.m` - Modernized splitting logic
- Validation checks (counts, Diss_IDs, class balance)
- Console and CSV demographics output

**Testing:** Verify split matches original counts and rules

---

### **WP4: Visualization Scripts**
**Duration:** Medium priority  
**Deliverables:**
- `visualize_preprocessing_comparison.m` - Before/after plots
- `visualize_preprocessing_steps.m` - Step-by-step effects
- `visualize_pca_framework.m` - PCA exploration plots

**Testing:** Visual inspection with multiple samples

---

### **WP5: Integration & Documentation**
**Duration:** Low priority  
**Deliverables:**
- Update `config.m` with preprocessing parameters
- Validation script: `validate_dataset_recreation.m`
- User documentation and examples
- Integration with existing pipeline stages

**Testing:** End-to-end pipeline run

---

## Technical Specifications

### Savitzky-Golay Filter Parameters

#### For Smoothing (PP2, Step 2):
```matlab
polynomial_order = 2;
window_size = 11;
derivative_order = 0;  % Smoothing only
smoothed = sgolayfilt(spectrum, polynomial_order, window_size);
```

#### For Second Derivative (Both PP1 & PP2):
```matlab
polynomial_order = 2;
window_size = 5;
derivative_order = 2;
second_deriv = sgolayfilt(spectrum, polynomial_order, window_size, derivative_order);
```

**Note:** MATLAB's `sgolayfilt` requires window_size to be odd and > polynomial_order

### Vector Normalization (L2 Norm)
```matlab
% For each spectrum (row vector)
spectrum_normalized = spectrum / norm(spectrum, 2);

% Verification: norm(spectrum_normalized, 2) should equal 1.0
```

### Binning Implementation
```matlab
% Bin factor = 4
% Average every 4 consecutive wavenumber points
bin_factor = 4;
n_bins = floor(length(spectrum) / bin_factor);
binned_spectrum = zeros(1, n_bins);
for i = 1:n_bins
    bin_start = (i-1) * bin_factor + 1;
    bin_end = i * bin_factor;
    binned_spectrum(i) = mean(spectrum(bin_start:bin_end));
end
```

---

## Success Criteria

### Validation Metrics
1. ✓ Dataset contains same number of samples as original
2. ✓ All Diss_IDs present in original are in new dataset
3. ✓ WHO-2 samples included in dataset
4. ✓ Train/Test split produces same counts as original
5. ✓ No patient overlap between train/test
6. ✓ Categorical variables properly formatted
7. ✓ CombinedRawSpectra matches input data
8. ✓ PP1 and PP2 spectra have expected dimensions
9. ✓ L2 norms of normalized spectra equal 1.0
10. ✓ Visualizations render without errors

### Performance Targets
- Data preparation: < 5 minutes for full dataset
- Memory usage: < 8 GB RAM
- No data loss during processing
- All preprocessing steps numerically stable (no NaN/Inf)

---

## Future Enhancements

### Phase 2 Additions
1. **Spectrum-level QC** with configurable thresholds
2. **Alternative representative spectra** (median, trimmed mean)
3. **Sample-level QC** with outlier detection
4. **Additional preprocessing approaches** (e.g., MSC, EMSC)
5. **Interactive visualizations** (plotly, HTML exports)
6. **Automated parameter optimization** for SG filters
7. **Batch processing** for very large datasets
8. **Export to other formats** (CSV, HDF5 for Python)

### Integration Points
- Quality control pipeline integration
- Feature selection with both PP1 and PP2
- Classifier comparison across preprocessing approaches
- Cross-validation framework updates

---

## Notes & Assumptions

1. **WHO-2 Samples:** Included in dataset, preprocessed with both approaches, but excluded from current train/test analysis (reserved for future project)

2. **Default Preprocessing:** PP1 is the standard for all downstream analysis unless explicitly specified otherwise

3. **Wavenumber Alignment:** After binning in PP2, wavenumber vector is also binned to maintain alignment

4. **Memory Efficiency:** Preprocessing done in-place where possible; large matrices not duplicated unnecessarily

5. **Error Handling:** All preprocessing functions include NaN/Inf checks and graceful degradation

6. **Reproducibility:** All random operations (train/test split) use fixed seed for reproducibility

---

## References

### Preprocessing Order Rationale
The BSNC sequence (Binning → Smoothing → Normalization → baseline Correction) is based on:
- Binning first to reduce dimensionality before computationally expensive operations
- Smoothing before normalization to avoid amplifying noise
- Normalization before baseline correction to standardize intensity scaling
- Baseline correction (2nd derivative) last to remove remaining systematic variations

### Savitzky-Golay Filter Theory
- Fits polynomial to sliding window
- Provides smooth derivatives without amplifying noise
- Window size determines smoothing degree (larger = smoother)
- Polynomial order affects fit quality (2 or 3 typical for spectra)

### L2 Normalization
- Also called "Euclidean normalization" or "unit vector normalization"
- Removes multiplicative scaling factors
- Focuses on spectral shape rather than absolute intensity
- Commonly used in chemometrics and spectroscopy

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-24 | GitHub Copilot | Initial implementation plan |

---

**End of Document**
