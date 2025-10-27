# Preprocessing Module

This module contains all spectral preprocessing functions for the meningioma FTIR classification pipeline.

## Overview

The preprocessing module implements two distinct preprocessing approaches (PP1 and PP2) for FTIR spectra, following the BSNC sequence: **B**inning → **S**moothing → **N**ormalization → baseline **C**orrection.

## Files

- **`apply_binning.m`** - Spectral binning (reduces resolution, improves SNR)
- **`apply_vector_normalization.m`** - L2 normalization (unit vector normalization)
- **`apply_sg_smoothing.m`** - Savitzky-Golay smoothing (noise reduction)
- **`apply_sg_derivative.m`** - Savitzky-Golay derivative (baseline correction)
- **`create_preprocessing_config.m`** - Configuration structure generator
- **`preprocess_spectra.m`** - Main orchestrator function
- **`test_preprocessing_functions.m`** - Test suite

## Quick Start

```matlab
% Add preprocessing directory to path
addpath('src/preprocessing');

% Load your data
load('wavenumbers.mat');  % wavenumber vector
raw_spectra = ...; % Your raw spectra matrix (rows = spectra, cols = wavenumbers)

% Preprocess with PP1 (standard, default for analysis)
[pp1_spectra, pp1_wavenumbers] = preprocess_spectra(raw_spectra, wavenumbers, 'PP1');

% Preprocess with PP2 (enhanced with binning/smoothing)
[pp2_spectra, pp2_wavenumbers] = preprocess_spectra(raw_spectra, wavenumbers, 'PP2');
```

## Preprocessing Approaches

### PP1 - Standard Pipeline ⭐ (Default)

**Purpose:** Minimal processing, maximum information retention

**Steps:**
1. ❌ No binning
2. ❌ No smoothing  
3. ✅ Vector normalization (L2 norm = 1)
4. ✅ Second derivative (SG: window=5, poly=2, deriv=2)

**Output:** Same resolution as input (441 wavenumbers)

**Use for:** Standard analysis, feature extraction, classification

---

### PP2 - Enhanced Pipeline

**Purpose:** Noise reduction through binning and smoothing

**Steps:**
1. ✅ Binning (factor 4) → reduces to ~110 wavenumbers
2. ✅ SG smoothing (window=11, poly=2, deriv=0)
3. ✅ Vector normalization (L2 norm = 1)
4. ✅ Second derivative (SG: window=5, poly=2, deriv=2)

**Output:** Reduced resolution (~110 wavenumbers)

**Use for:** Noisy data, exploratory analysis, comparison studies

## Function Reference

### `preprocess_spectra` - Main Function

**Syntax:**
```matlab
[processed_spectra, processed_wavenumbers] = preprocess_spectra(spectra, wavenumbers, config)
```

**Inputs:**
- `spectra` - Matrix (rows = spectra, cols = wavenumbers)
- `wavenumbers` - Row vector of wavenumber values
- `config` - Either:
  - String: `'PP1'` or `'PP2'`
  - Struct: from `create_preprocessing_config()`

**Outputs:**
- `processed_spectra` - Preprocessed spectra matrix
- `processed_wavenumbers` - Wavenumber vector (may be binned)

**Example:**
```matlab
[processed, wn] = preprocess_spectra(raw, wavenumbers, 'PP1');
```

---

### `create_preprocessing_config` - Configuration

**Syntax:**
```matlab
config = create_preprocessing_config(approach)
```

**Inputs:**
- `approach` - `'PP1'`, `'PP2'`, or `'custom'`

**Output:**
- `config` - Structure with all preprocessing parameters

**Example:**
```matlab
% Get PP1 config
cfg = create_preprocessing_config('PP1');

% Customize
cfg = create_preprocessing_config('custom');
cfg.apply_binning = true;
cfg.bin_factor = 2;
[processed, wn] = preprocess_spectra(raw, wavenumbers, cfg);
```

---

### `apply_binning` - Spectral Binning

**Syntax:**
```matlab
[binned_spectra, binned_wavenumbers] = apply_binning(spectra, wavenumbers, bin_factor)
```

**Purpose:** Reduce spectral resolution by averaging neighboring points

**Parameters:**
- `bin_factor` - Integer (default: 4)

**Effect:** Reduces 441 points → ~110 points (with factor 4)

---

### `apply_vector_normalization` - L2 Normalization

**Syntax:**
```matlab
normalized_spectra = apply_vector_normalization(spectra)
```

**Purpose:** Normalize each spectrum to unit length (L2 norm = 1)

**Formula:** `normalized = spectrum / norm(spectrum, 2)`

**Effect:** Removes intensity scaling, focuses on spectral shape

---

### `apply_sg_smoothing` - Savitzky-Golay Smoothing

**Syntax:**
```matlab
smoothed_spectra = apply_sg_smoothing(spectra, window_size, polynomial_order)
```

**Purpose:** Reduce high-frequency noise while preserving peaks

**Parameters:**
- `window_size` - Odd integer (default: 11)
- `polynomial_order` - Integer (default: 2)

**Note:** This is smoothing only (derivative order = 0)

---

### `apply_sg_derivative` - Savitzky-Golay Derivative

**Syntax:**
```matlab
derivative_spectra = apply_sg_derivative(spectra, window_size, polynomial_order, derivative_order)
```

**Purpose:** Compute derivatives for baseline correction

**Parameters:**
- `window_size` - Odd integer (default: 5)
- `polynomial_order` - Integer (default: 2)
- `derivative_order` - Integer (default: 2 for second derivative)

**Effect:** Removes baseline drift, enhances peak resolution

## Testing

Run the test suite to verify all functions:

```matlab
cd src/preprocessing
test_preprocessing_functions
```

The test suite includes:
- Synthetic data generation
- Individual function tests
- Full pipeline tests (PP1 and PP2)
- Edge case handling
- Visual comparison plots

## Technical Notes

### L2 Normalization vs SNV

**L2 Normalization (implemented):**
- Formula: `spectrum / norm(spectrum, 2)`
- Result: Unit vector (||spectrum|| = 1)

**SNV (not implemented):**
- Formula: `(spectrum - mean(spectrum)) / std(spectrum)`
- Result: Mean = 0, Std = 1

### Savitzky-Golay Filter

The SG filter fits a polynomial to a sliding window and either:
- Returns the fitted value (smoothing, deriv=0)
- Returns the analytical derivative (baseline correction, deriv=1 or 2)

**Constraints:**
- `window_size` must be odd
- `window_size > polynomial_order`
- `derivative_order <= polynomial_order`

### Binning

Averages every `N` consecutive wavenumber points. If spectrum length is not divisible by `N`, trailing points are excluded with a warning.

## Integration with Pipeline

These functions are used by:
- `prepare_ftir_dataset.m` - Main data preparation script
- Visualization scripts
- Quality control analysis

## References

See `docs/PREPROCESSING_IMPLEMENTATION_PLAN.md` for detailed specifications and rationale.

---

**Author:** GitHub Copilot  
**Date:** 2025-10-24  
**Version:** 1.0
