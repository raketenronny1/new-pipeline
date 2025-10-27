# **MENINGIOMA FT-IR CLASSIFICATION PIPELINE WITH INTEGRATED QUALITY CONTROL**

## **CRITICAL CONTEXT UNDERSTANDING**

Before you begin coding, confirm you understand these key points:

### **Ask me to verify your understanding:**

1. **Data Structure**: The input files in `data/` subfolder contain:
    - `data_table_train.mat` with variable `dataTableTrain` (MATLAB table, ~44 probes)
    - `data_table_test.mat` with variable `dataTableTest` (MATLAB table, ~32 probes)
    - `wavenumbers.mat` with variable `wavenumbers_roi` (wavenumber vector)
    - Each table row = one probe/patient with:
        - `Diss_ID` (cell array of strings): Patient identifier
        - `Patient_ID` (string): Patient ID
        - `Fall_ID` (double): Case ID
        - `WHO_Grade` (categorical): 'WHO-1' or 'WHO-3'
        - `Sex`, `Age`: Demographics
        - `CombinedSpectra` (cell): Each cell contains a matrix (N_spectra × N_wavenumbers) of preprocessed spectra for that probe

2. **Train/Test Split Philosophy**:
    - Training set (44 probes): Used for ALL model development, feature selection, hyperparameter tuning via cross-validation
    - Test set (32 probes): **TOUCHED EXACTLY ONCE** at the very end for unbiased performance estimation
    - **NO DATA LEAKAGE**: All preprocessing parameters, PCA loadings, feature selection must be learned ONLY from training set
    - **QC Philosophy**: Quality control ensures data integrity and can use fixed thresholds applied to both sets, but any data-driven QC parameters (e.g., PCA for outlier detection) must be learned from training set only

3. **Cross-Validation Strategy**:
    - Use **simple stratified K-fold CV** (not nested CV) on training set only
    - Why simple? Because we have a separate held-out test set
    - CV on training set → select best model → train final model on all training samples → evaluate once on test set

4. **Performance Metrics**:
    - Primary: Balanced Accuracy, F2-score for WHO-3
    - Secondary: Sensitivity (WHO-3), Specificity (WHO-1), PPV, NPV, F1, AUC-ROC
    - WHO-3 is positive class (more important to catch high-grade tumors)

**DO YOU UNDERSTAND? If yes, proceed to implementation. If no, ask clarifying questions.**

---

## **IMPLEMENTATION REQUIREMENTS**

### **Platform & Environment**

- MATLAB R2025b
- All toolboxes available (Statistics & ML, Signal Processing)
- Working directory: Project root (where `data/`, `models/`, `results/` folders exist)
- Code must be well-commented, modular, and production-ready

### **Required Folder Structure**

```
project_root/
├── data/
│   ├── data_table_train.mat
│   ├── data_table_test.mat
│   └── wavenumbers.mat
├── models/
│   └── meningioma_ftir_pipeline/  (create this)
├── results/
│   └── meningioma_ftir_pipeline/  (create this)
│       └── qc/  (create for QC outputs)
└── src/
    └── meningioma_ftir_pipeline/  (contains all code)
```

---

## **PHASE 0: QUALITY CONTROL AND DATA VALIDATION**

### **Script: `quality_control_analysis.m`**

**Philosophy**: Quality control ensures data integrity and removes technical artifacts. QC uses fixed, literature-based thresholds that are applied identically to training and test sets. This is distinct from modeling parameters (like PCA loadings) which must be learned only from training data.

**Inputs**:
- `data/data_table_train.mat`
- `data/data_table_test.mat`
- `data/wavenumbers.mat`

**QC Strategy**: Multi-level approach from individual spectra to sample-level assessment

---

### **LEVEL 1: SPECTRUM-LEVEL QUALITY CONTROL**

Apply these filters to EACH of the ~768 spectra per sample:

#### **1.1 Signal-to-Noise Ratio (SNR)**

```matlab
% For each spectrum:
% Signal region: Peak area (1000-1700 cm⁻¹) - contains Amide I, II, biological peaks
signal_region_idx = find(wavenumbers >= 1000 & wavenumbers <= 1700);
signal = max(spectrum(signal_region_idx)) - min(spectrum(signal_region_idx));

% Noise region: Flat baseline region (1750-1800 cm⁻¹) - minimal biological features
noise_region_idx = find(wavenumbers >= 1750 & wavenumbers <= 1800);
noise = std(spectrum(noise_region_idx));

SNR = signal / noise;

% Threshold: SNR < 10 indicates poor quality
% Literature basis: Standard spectroscopy QC threshold
```

**Action**: Flag spectra with SNR < 10 for exclusion  
**Expected impact**: Typically excludes 5-15% of spectra per sample

#### **1.2 Saturation Check**

```matlab
% After baseline correction, absorbance should be in 0-2.0 range
% Detector saturation occurs at high absorbance values
max_absorbance = max(spectrum);

% Threshold: Absorbance > 1.8 indicates saturation risk
% Values > 2.0 indicate definite saturation (non-linear detector response)
```

**Action**: Flag spectra with max absorbance > 1.8 for exclusion  
**Rationale**: Saturated spectra are non-linear and unreliable for quantitative analysis

#### **1.3 Baseline Quality Assessment**

```matlab
% Even after preprocessing, check if baseline is adequately flat
% Evaluate standard deviation in non-peak regions

% Region 1: 950-1000 cm⁻¹ (if available in your ROI)
% Region 2: 1750-1800 cm⁻¹
baseline_region_1 = find(wavenumbers >= 950 & wavenumbers <= 1000);
baseline_region_2 = find(wavenumbers >= 1750 & wavenumbers <= 1800);

baseline_sd = std([spectrum(baseline_region_1); spectrum(baseline_region_2)]);

% Threshold: SD > 0.02 absorbance units indicates poor baseline correction
```

**Action**: Flag spectra with baseline SD > 0.02 for exclusion  
**Causes of failure**: Mie scattering artifacts, edge effects, mounting problems

#### **1.4 Negative Absorbance Check**

```matlab
% After proper baseline correction, minimal negative values expected
min_absorbance = min(spectrum);
negative_points = sum(spectrum < -0.1);

% Threshold: Absorbance < -0.1 or >5% negative points indicates problems
```

**Action**: Flag spectra with significant negative absorbance for exclusion  
**Causes**: Over-correction, instrument artifacts, sample mounting issues

#### **1.5 Biological Plausibility - Peak Presence**

```matlab
% All tissue spectra MUST show characteristic biological peaks:
% Amide I peak: ~1650 cm⁻¹ (strongest protein feature)
% Amide II peak: ~1550 cm⁻¹ (protein feature)

amide_I_region = find(wavenumbers >= 1630 & wavenumbers <= 1670);
amide_II_region = find(wavenumbers >= 1530 & wavenumbers <= 1570);

amide_I_height = max(spectrum(amide_I_region));
amide_II_height = max(spectrum(amide_II_region));

% Peak ratio should be 1.5-3.0 for protein-containing tissue
peak_ratio = amide_I_height / amide_II_height;

% Threshold: Ratio outside 1.2-3.5 range indicates non-tissue material
```

**Action**: Flag spectra with anomalous peak ratios for exclusion  
**Indicates**: Contamination, non-biological material, acquisition failure

---

### **LEVEL 2: SAMPLE-LEVEL QUALITY CONTROL**

After Level 1 filtering, assess each sample (probe) as a whole:

#### **2.1 Within-Sample Spectral Consistency**

```matlab
% For each sample, calculate pairwise correlations between all remaining spectra
% High correlation = homogeneous sampling of consistent tissue

for i = 1:n_spectra
    for j = i+1:n_spectra
        corr_matrix(i,j) = corr(spectra(i,:)', spectra(j,:)');
    end
end

mean_within_sample_correlation = mean(corr_matrix(triu(true(size(corr_matrix)),1)));

% Threshold: Mean correlation < 0.85 suggests heterogeneous/problematic sample
```

**Interpretation**:
- **High correlation (R > 0.90)**: Homogeneous tissue, good sampling
- **Moderate correlation (0.85-0.90)**: Acceptable variability
- **Low correlation (< 0.85)**: Heterogeneous sampling (tumor + necrosis + normal), contamination, or multiple tissue types

**Action**: Flag entire sample if mean correlation < 0.85

#### **2.2 Minimum Spectra Threshold**

```matlab
% After Level 1 exclusions, count remaining high-quality spectra per sample
n_valid_spectra = sum(~flagged_spectra);

% Threshold: Minimum 100 valid spectra per sample (13% retention rate)
% This ensures robust representative spectrum calculation
```

**Action**: Consider excluding entire sample if < 100 valid spectra remain  
**Rationale**: Insufficient data to reliably characterize the sample  
**Trade-off**: May reduce sample size from 44 → 40-42, but improves data quality

#### **2.3 Representative Spectrum Calculation**

**Recommendation for this 44-sample study: Quality-Filtered Mean**

```matlab
% Step 1: Apply Level 1 QC filters
valid_spectra_mask = (SNR >= 10) & ...
                     (max_absorbance <= 1.8) & ...
                     (baseline_sd <= 0.02) & ...
                     (min_absorbance >= -0.1) & ...
                     (peak_ratio >= 1.2 & peak_ratio <= 3.5);

% Step 2: Among valid spectra, remove extreme outliers using Mahalanobis distance
% Compute within-sample PCA (using valid spectra only)
[coeff_sample, score_sample] = pca(valid_spectra);

% Use first 5-10 PCs for outlier detection
n_pcs_outlier = min(10, size(score_sample, 2));
scores_for_outlier = score_sample(:, 1:n_pcs_outlier);

% Mahalanobis distance in PC space
mahal_dist = mahal(scores_for_outlier, scores_for_outlier);

% Chi-squared threshold at 99% confidence
chi2_threshold = chi2inv(0.99, n_pcs_outlier);

% Exclude top 1% most extreme outliers
outlier_mask = mahal_dist > chi2_threshold;

% Step 3: Calculate quality-filtered mean
final_valid_spectra = valid_spectra_mask & ~outlier_mask;
representative_spectrum = mean(spectra(final_valid_spectra, :), 1);

% Also compute median for sensitivity analysis (report in supplementary)
representative_spectrum_median = median(spectra(final_valid_spectra, :), 1);
```

**Justification**:
- **Maximizes SNR**: With only 44 samples, every bit of signal matters
- **Removes artifacts**: Prevents technical outliers from degrading representative spectrum
- **Standard in clinical studies**: Published diagnostic studies use this approach
- **Expected retention**: Average 612 ± 45 spectra per sample (range: 520-680)

---

### **LEVEL 3: CROSS-SAMPLE OUTLIER DETECTION**

**CRITICAL**: This step must respect train/test separation to avoid leakage

#### **3.1 Mahalanobis Distance-Based Outlier Detection**

**Approach for small sample size (44 training samples)**:

```matlab
% IMPORTANT: Perform separately for training set only
% Do NOT use test set in this calculation

% Step 1: Create representative spectra from QC'd data (from Level 2)
% X_train_representatives: [44 samples × N_wavenumbers]

% Step 2: Compute PCA on training representative spectra
[coeff_train, score_train, latent_train] = pca(X_train_representatives);

% Use first 5-10 PCs (sufficient variance, invertible covariance)
n_pcs_outlier = min(10, sum(cumsum(latent_train)/sum(latent_train) <= 0.95));

% Step 3: Calculate Mahalanobis distance SEPARATELY for each WHO grade
% This prevents flagging samples just because they're from different biological class

% WHO-1 group
idx_WHO1 = find(y_train == 'WHO-1');
scores_WHO1 = score_train(idx_WHO1, 1:n_pcs_outlier);
mahal_WHO1 = mahal(scores_WHO1, scores_WHO1);

% WHO-3 group
idx_WHO3 = find(y_train == 'WHO-3');
scores_WHO3 = score_train(idx_WHO3, 1:n_pcs_outlier);
mahal_WHO3 = mahal(scores_WHO3, scores_WHO3);

% Step 4: Determine outliers using conservative threshold
% 99% confidence (more conservative than 95% to avoid over-exclusion)
chi2_threshold = chi2inv(0.99, n_pcs_outlier);

outliers_WHO1 = mahal_WHO1 > chi2_threshold;
outliers_WHO3 = mahal_WHO3 > chi2_threshold;
```

**Decision Rules**:
1. **Maximum exclusions**: Remove at most 2 samples total (1 from each group)
2. **Manual inspection**: Visually inspect spectra of flagged samples before exclusion
3. **Documentation**: Record which samples flagged, which excluded, and justification
4. **Transparency**: Report in methods how many samples flagged vs. actually excluded

**Interpretation**:
- **Outlier in WHO-1**: Sample spectrally distant from other WHO-1 samples
- **Possible causes**: Labeling error, unusual biology, technical problem, mixed tissue
- **Action**: Investigate metadata (age, sex, subtype), re-examine spectrum, decide on exclusion

#### **3.2 Test Set Evaluation** (Informational Only)

```matlab
% For test set: Apply same PCA transformation learned from training set
% This checks if test samples are within training distribution

% Center test representatives using training mean
X_test_representatives_centered = X_test_representatives - mean(X_train_representatives);

% Project using training PC loadings
score_test = X_test_representatives_centered * coeff_train(:, 1:n_pcs_outlier);

% Calculate distance to training distribution (informational)
% Use training scores to define "normal" range
mahal_test = mahal(score_test, score_train(:, 1:n_pcs_outlier));

% Do NOT exclude test samples based on this
% But report if test set contains extreme outliers relative to training
```

**Purpose**: Assess domain shift between train and test sets  
**Action**: If test set contains many extreme outliers, report this as limitation

---

### **LEVEL 4: BIOLOGICAL PLAUSIBILITY & BATCH EFFECTS**

#### **4.1 Expected WHO Grade Differences**

```matlab
% Verify that WHO-1 and WHO-3 groups show spectral differences
% Perform PCA on training representative spectra, color by WHO grade

% Expected biological differences (literature-based):
% - WHO-3: Increased nucleic acid bands (1000-1250 cm⁻¹) due to proliferation
% - WHO-3: Altered Amide I position/shape (protein secondary structure changes)
% - WHO-3: Changed lipid/protein ratios

% Quality check: Groups should show some separation in PCA
% If completely overlapping, suggests:
%   - Labeling errors
%   - Insufficient biological difference
%   - Technical variability overwhelming signal
```

**Action**: Create PCA plot (PC1 vs PC2) colored by WHO grade for QC report

#### **4.2 Batch Effect Assessment**

**Temporal batches** (if metadata available):
```matlab
% Check if acquisition date influences spectra
% PCA colored by acquisition date should show no clustering

% If temporal batches exist:
%   - Include date as covariate, OR
%   - Exclude problematic batches, OR
%   - Apply batch correction (e.g., ComBat)
```

**Sample preparation batches**:
```matlab
% Verify consistent preparation:
%   - Same fixation protocol
%   - Same section thickness (critical: 4μm vs 8μm affects spectra)
%   - Same mounting medium

% Check metadata for thickness variability
% If variable: Either exclude or model as covariate
```

**Operator effects**:
```matlab
% If multiple operators acquired spectra:
% PCA colored by operator should show no separation
```

---

### **QC OUTPUTS**

**Script should generate**:

1. **QC Report Document** (`qc_report.pdf`):
   - Summary statistics: Spectra excluded per sample (Level 1)
   - Samples flagged/excluded (Level 2-3)
   - Justification for all exclusions
   - Retention rates: Mean spectra per sample after QC

2. **QC Metrics Table** (`qc_metrics_summary.csv`):
   ```
   Sample_ID | WHO_Grade | n_Original | n_After_SNR | n_After_Saturation | n_After_Baseline | n_After_Amide | n_Final | Within_Corr | Outlier_Flag | Excluded
   ```

3. **Visualizations** (save to `results/meningioma_ftir_pipeline/qc/`):
   - `qc_snr_distribution.png`: Histogram of SNR values across all spectra
   - `qc_correlation_boxplot.png`: Within-sample correlation distributions
   - `qc_spectra_retention.png`: Bar chart of retained spectra per sample
   - `qc_pca_outliers.png`: PCA plot with flagged outliers highlighted
   - `qc_who_grade_separation.png`: PCA colored by WHO grade (sanity check)

4. **Cleaned Data Files**:
   - `results/meningioma_ftir_pipeline/qc/cleaned_train_representatives.mat`: QC'd training representative spectra [n_samples × n_wavenumbers]
   - `results/meningioma_ftir_pipeline/qc/cleaned_test_representatives.mat`: QC'd test representative spectra
   - `results/meningioma_ftir_pipeline/qc/qc_flags.mat`: Detailed flags for all samples/spectra

5. **Sensitivity Analysis**:
   - `results/meningioma_ftir_pipeline/qc/median_representatives.mat`: Median-based representatives for comparison

---

## **PHASE 1: DATA LOADING & INTEGRATION WITH QC**

### **Script: `load_and_prepare_data.m`**

**Inputs**:
- `data/data_table_train.mat`
- `data/data_table_test.mat`
- `data/wavenumbers.mat`
- QC results from Phase 0

**Processing Steps**:

1. **Load Raw Data**
```matlab
% Load training data
load('data/data_table_train.mat', 'dataTableTrain');
load('data/data_table_test.mat', 'dataTableTest');
load('data/wavenumbers.mat', 'wavenumbers_roi');
```

2. **Load QC Results**
```matlab
% Load QC-cleaned representative spectra (one per sample)
load('results/meningioma_ftir_pipeline/qc/cleaned_train_representatives.mat');
load('results/meningioma_ftir_pipeline/qc/cleaned_test_representatives.mat');
load('results/meningioma_ftir_pipeline/qc/qc_flags.mat');

% Remove any samples flagged for exclusion
samples_to_keep_train = ~qc_flags.train_excluded;
samples_to_keep_test = ~qc_flags.test_excluded;
```

3. **Create Analysis-Ready Datasets**
```matlab
% Training set
X_train = cleaned_train_representatives(samples_to_keep_train, :);
y_train = dataTableTrain.WHO_Grade(samples_to_keep_train);
probe_ids_train = dataTableTrain.Diss_ID(samples_to_keep_train);

% Test set  
X_test = cleaned_test_representatives(samples_to_keep_test, :);
y_test = dataTableTest.WHO_Grade(samples_to_keep_test);
probe_ids_test = dataTableTest.Diss_ID(samples_to_keep_test);
```

4. **Final Quality Checks**
```matlab
% Verify dimensions
fprintf('Training set: %d samples × %d wavenumbers\n', size(X_train));
fprintf('Test set: %d samples × %d wavenumbers\n', size(X_test));

% Check class balance
fprintf('Training WHO-1: %d, WHO-3: %d\n', sum(y_train=='WHO-1'), sum(y_train=='WHO-3'));
fprintf('Test WHO-1: %d, WHO-3: %d\n', sum(y_test=='WHO-1'), sum(y_test=='WHO-3'));

% Verify no NaN or Inf
assert(~any(isnan(X_train(:)) | isinf(X_train(:))), 'Training data contains NaN/Inf');
assert(~any(isnan(X_test(:)) | isinf(X_test(:))), 'Test data contains NaN/Inf');
```

5. **Outputs**:
```matlab
% Save analysis-ready data
trainingData = struct();
trainingData.X = X_train;
trainingData.y = y_train;
trainingData.probe_ids = probe_ids_train;
trainingData.probe_table = dataTableTrain(samples_to_keep_train, :);

testData = struct();
testData.X = X_test;
testData.y = y_test;
testData.probe_ids = probe_ids_test;
testData.probe_table = dataTableTest(samples_to_keep_test, :);

save('results/meningioma_ftir_pipeline/preprocessed_data.mat', 'trainingData', 'testData', 'wavenumbers_roi');
```

---

## **PHASE 2: FEATURE SELECTION ON TRAINING SET ONLY**

### **Script: `perform_feature_selection.m`**

**Input**: `trainingData` from Phase 1 (QC-cleaned representative spectra)

**Feature Selection Method**: PCA (dimensionality reduction)

**CRITICAL**: PCA computed ONLY on training set, then applied to test set

**Processing Steps**:

1. **Compute PCA on Training Data**
```matlab
% Load QC-cleaned training data
load('results/Phase3_v2/preprocessed_data.mat', 'trainingData');

% Compute PCA (MATLAB auto-centers data)
[coeff, score, latent, ~, explained, mu] = pca(trainingData.X);

% Note: 'mu' is the mean of training data, needed for test set transformation
```

2. **Select Number of Components**
```matlab
% Strategy 1: Cumulative variance threshold (95-99%)
variance_threshold = 0.95;
n_components = find(cumsum(explained) >= variance_threshold * 100, 1);

% Strategy 2: Scree plot elbow method (visual inspection)
% Typically results in 5-10 components for spectroscopic data

fprintf('Selected %d PCs explaining %.1f%% variance\n', ...
        n_components, sum(explained(1:n_components)));

% Create variance explained plot
figure;
subplot(1,2,1);
plot(1:min(20, length(explained)), explained(1:min(20, length(explained))), 'bo-');
xlabel('Principal Component'); ylabel('Variance Explained (%)');
title('Scree Plot');
grid on;

subplot(1,2,2);
plot(1:min(20, length(explained)), cumsum(explained(1:min(20, length(explained)))), 'ro-');
hold on; yline(95, 'k--', '95%'); yline(99, 'k--', '99%');
xlabel('Principal Component'); ylabel('Cumulative Variance (%)');
title('Cumulative Variance');
grid on;

saveas(gcf, 'results/Phase3_v2/pca_variance_explained.png');
```

3. **Save PCA Model**
```matlab
% This model will be used to transform test set
pca_model = struct();
pca_model.coeff = coeff;  % PC loadings [n_wavenumbers × n_components]
pca_model.mu = mu;  % Training data mean [1 × n_wavenumbers]
pca_model.latent = latent;  % Eigenvalues
pca_model.explained = explained;  % Variance explained per PC
pca_model.n_components = n_components;  % Number selected

save('models/meningioma_ftir_pipeline/pca_model.mat', 'pca_model');
```

4. **Transform Training Data**
```matlab
% Extract selected components
X_train_pca = score(:, 1:n_components);

% Save transformed training data
save('results/Phase3_v2/X_train_pca.mat', 'X_train_pca');
```

5. **Visualize PCA Space**
```matlab
% Create 2D PCA plot colored by WHO grade
figure;
scatter(X_train_pca(trainingData.y=='WHO-1', 1), ...
        X_train_pca(trainingData.y=='WHO-1', 2), ...
        100, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
scatter(X_train_pca(trainingData.y=='WHO-3', 1), ...
        X_train_pca(trainingData.y=='WHO-3', 2), ...
        100, 'r', 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('PC1'); ylabel('PC2');
legend({'WHO-1', 'WHO-3'}, 'Location', 'best');
title('Training Data in PCA Space');
grid on;
saveas(gcf, 'results/meningioma_ftir_pipeline/pca_training_space.png');
```

**Outputs**:
- `models/meningioma_ftir_pipeline/pca_model.mat`: PCA transformation parameters
- `results/meningioma_ftir_pipeline/X_train_pca.mat`: Transformed training data
- `results/meningioma_ftir_pipeline/pca_variance_explained.png`: Variance plots
- `results/meningioma_ftir_pipeline/pca_training_space.png`: 2D visualization

---

## **PHASE 3: MODEL SELECTION VIA CROSS-VALIDATION (TRAINING SET ONLY)**

### **Script: `run_cross_validation.m`**

**Input**: `X_train_pca`, `trainingData.y` (from Phase 2)

**CV Strategy**: Stratified 5-Fold CV, repeated 50 times

**Classifiers to Test**:
1. Linear Discriminant Analysis (LDA)
2. Partial Least Squares Discriminant Analysis (PLS-DA)
3. Support Vector Machine (SVM) with RBF kernel
4. Random Forest

**Processing Steps**:

1. **Set Up CV Partitioning**
```matlab
% Load transformed training data
load('results/meningioma_ftir_pipeline/X_train_pca.mat', 'X_train_pca');
load('results/meningioma_ftir_pipeline/preprocessed_data.mat', 'trainingData');

% Set random seed for reproducibility
rng(42);

% Create stratified CV partition
% At sample level (not spectrum level, since we have representative spectra)
n_folds = 5;
n_repeats = 50;

% Store results
cv_results = cell(n_repeats, 1);
```

2. **Define Classifiers and Hyperparameter Grids**
```matlab
% LDA: No hyperparameters (use default)
classifier_lda = @(X, y) fitcdiscr(X, y, 'DiscrimType', 'linear');

% PLS-DA: Optimize number of components
plsda_n_comp_grid = 1:min(10, size(X_train_pca, 2));

% SVM: Grid search C and gamma
svm_C_grid = 10.^(-2:0.5:2);  % [0.01, 0.03, ..., 100]
svm_gamma_grid = 10.^(-3:0.5:1);  % [0.001, 0.003, ..., 10]

% Random Forest: Optimize n_trees and max_depth
rf_n_trees_grid = [50, 100, 200, 500];
rf_max_depth_grid = [5, 10, 20, 30];
```

3. **Run Cross-Validation Loop**
```matlab
for rep = 1:n_repeats
    % Create CV partition for this repeat
    cv_partition = cvpartition(trainingData.y, 'KFold', n_folds);
    
    fold_results = struct();
    fold_results.predictions = cell(n_folds, 1);
    fold_results.scores = cell(n_folds, 1);
    fold_results.true_labels = cell(n_folds, 1);
    
    for fold = 1:n_folds
        % Get train/validation indices for this fold
        train_idx = training(cv_partition, fold);
        val_idx = test(cv_partition, fold);
        
        X_train_fold = X_train_pca(train_idx, :);
        y_train_fold = trainingData.y(train_idx);
        X_val_fold = X_train_pca(val_idx, :);
        y_val_fold = trainingData.y(val_idx);
        
        % Train and evaluate each classifier
        % [Detailed implementation for each classifier with hyperparameter tuning]
        
        % Store fold predictions
        fold_results.predictions{fold} = predictions;
        fold_results.scores{fold} = scores;
        fold_results.true_labels{fold} = y_val_fold;
    end
    
    % Calculate metrics for this repeat
    [accuracy, sensitivity, specificity, ppv, npv, f1, f2, auc] = ...
        calculate_metrics(fold_results);
    
    cv_results{rep}.accuracy = accuracy;
    cv_results{rep}.sensitivity = sensitivity;
    cv_results{rep}.specificity = specificity;
    cv_results{rep}.f2 = f2;
    % ... store all metrics
end
```

4. **Aggregate CV Results**
```matlab
% Calculate mean ± SD across repeats for each classifier
summary_table = table();
summary_table.Classifier = categorical({'LDA', 'PLSDA', 'SVM', 'RandomForest'})';

for clf = 1:4
    clf_results = cv_results(clf, :);
    
    summary_table.Mean_Accuracy(clf) = mean([clf_results.accuracy]);
    summary_table.SD_Accuracy(clf) = std([clf_results.accuracy]);
    summary_table.Mean_Sensitivity_WHO3(clf) = mean([clf_results.sensitivity]);
    summary_table.SD_Sensitivity_WHO3(clf) = std([clf_results.sensitivity]);
    % ... all other metrics
    
    % Calculate 95% confidence intervals
    summary_table.CI95_F2_lower(clf) = prctile([clf_results.f2], 2.5);
    summary_table.CI95_F2_upper(clf) = prctile([clf_results.f2], 97.5);
end

% Save results
writetable(summary_table, 'results/meningioma_ftir_pipeline/cv_performance.csv');
```

5. **Select Best Classifier**
```matlab
% Primary criterion: Highest Mean F2-score for WHO-3 detection
[~, best_idx] = max(summary_table.Mean_F2_WHO3);
best_classifier = summary_table.Classifier(best_idx);

fprintf('\nBest classifier: %s\n', char(best_classifier));
fprintf('Mean F2-score: %.3f ± %.3f\n', ...
        summary_table.Mean_F2_WHO3(best_idx), ...
        summary_table.SD_F2_WHO3(best_idx));

% Save best classifier selection
best_model_info = struct();
best_model_info.classifier = char(best_classifier);
best_model_info.cv_performance = summary_table(best_idx, :);
save('results/meningioma_ftir_pipeline/best_classifier_selection.mat', 'best_model_info');
```

**Outputs**:
- `results/meningioma_ftir_pipeline/cv_performance.csv`: Full CV results table
- `results/meningioma_ftir_pipeline/best_classifier_selection.mat`: Selected classifier info
- `results/meningioma_ftir_pipeline/cv_boxplots.png`: Performance metric distributions

---

## **PHASE 4: TRAIN FINAL MODEL ON ALL TRAINING DATA**

### **Script: `train_final_model.m`**

**Input**: Best classifier from Phase 3, all training data (42-44 samples after QC)

**Processing Steps**:

1. **Load Best Classifier Info and Optimal Hyperparameters**
```matlab
load('results/Phase3_v2/best_classifier_selection.mat', 'best_model_info');
load('results/Phase3_v2/X_train_pca.mat', 'X_train_pca');
load('results/Phase3_v2/preprocessed_data.mat', 'trainingData');

% Set random seed
rng(42);
```

2. **Train on ALL Training Data**
```matlab
% Use optimal hyperparameters determined from CV
% Train on complete X_train_pca with all training samples

switch best_model_info.classifier
    case 'LDA'
        final_model = fitcdiscr(X_train_pca, trainingData.y, ...
                                'DiscrimType', 'linear');
        
    case 'PLSDA'
        final_model = fitcpls(X_train_pca, trainingData.y, ...
                              'NumComponents', best_model_info.optimal_n_comp);
        
    case 'SVM'
        final_model = fitcsvm(X_train_pca, trainingData.y, ...
                              'KernelFunction', 'rbf', ...
                              'BoxConstraint', best_model_info.optimal_C, ...
                              'KernelScale', best_model_info.optimal_gamma);
        
    case 'RandomForest'
        final_model = TreeBagger(best_model_info.optimal_n_trees, ...
                                 X_train_pca, trainingData.y, ...
                                 'Method', 'classification', ...
                                 'MaxNumSplits', best_model_info.optimal_max_depth);
end
```

3. **Save Final Model with Metadata**
```matlab
final_model_package = struct();
final_model_package.model = final_model;
final_model_package.classifier_type = best_model_info.classifier;
final_model_package.hyperparameters = best_model_info.optimal_params;
final_model_package.training_date = datestr(now);
final_model_package.n_training_samples = size(X_train_pca, 1);
final_model_package.cv_performance = best_model_info.cv_performance;

save('models/meningioma_ftir_pipeline/final_model.mat', 'final_model_package');
```

**Outputs**:
- `models/meningioma_ftir_pipeline/final_model.mat`: Trained model with metadata

---

## **PHASE 5: TEST SET EVALUATION (DONE EXACTLY ONCE)**

### **Script: `evaluate_test_set.m`**

**Input**: Test data, PCA model, final trained model

**CRITICAL RULES**:
- Test set is used EXACTLY ONCE
- No iterative adjustments based on test performance
- No cherry-picking of results
- Report ALL metrics, not just favorable ones

**Processing Steps**:

1. **Load Models and Test Data**
```matlab
load('models/Phase3_v2/pca_model.mat', 'pca_model');
load('models/Phase3_v2/final_model.mat', 'final_model_package');
load('results/Phase3_v2/preprocessed_data.mat', 'testData');
```

2. **Apply PCA Transformation to Test Set**
```matlab
% CRITICAL: Use training-derived PCA parameters

% Center test data using TRAINING mean
X_test_centered = testData.X - pca_model.mu;

% Project using TRAINING PC loadings
X_test_pca = X_test_centered * pca_model.coeff(:, 1:pca_model.n_components);

fprintf('Test set transformed: %d samples × %d PCs\n', size(X_test_pca));
```

3. **Make Predictions**
```matlab
% Get predictions and probability scores
[y_pred, scores] = predict(final_model_package.model, X_test_pca);

% For TreeBagger (Random Forest), handle differently
if strcmp(final_model_package.classifier_type, 'RandomForest')
    [y_pred, scores] = predict(final_model_package.model, X_test_pca);
    % Convert cell array to categorical if needed
end
```

4. **Calculate ALL Performance Metrics**
```matlab
% Confusion matrix
cm = confusionmat(testData.y, y_pred);

% Accuracy metrics
accuracy = (cm(1,1) + cm(2,2)) / sum(cm(:));
balanced_accuracy = mean([cm(1,1)/sum(cm(1,:)), cm(2,2)/sum(cm(2,:))]);

% Sensitivity and Specificity
% Assuming WHO-3 is positive class (row 2, col 2)
sensitivity_WHO3 = cm(2,2) / sum(cm(2,:));  % True positive rate
specificity_WHO1 = cm(1,1) / sum(cm(1,:));  % True negative rate

% Predictive values
ppv = cm(2,2) / sum(cm(:,2));  % Positive predictive value
npv = cm(1,1) / sum(cm(:,1));  % Negative predictive value

% F-scores
precision = ppv;
recall = sensitivity_WHO3;
f1 = 2 * (precision * recall) / (precision + recall);

% F2-score (emphasizes recall over precision)
beta = 2;
f2 = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall);

% AUC-ROC
% Assuming scores(:,2) corresponds to WHO-3 probability
[X_roc, Y_roc, T, AUC] = perfcurve(testData.y, scores(:,2), 'WHO-3');
```

5. **Create Visualizations**
```matlab
% --- Confusion Matrix Heatmap ---
figure('Position', [100, 100, 800, 600]);
h = heatmap({'Predicted WHO-1', 'Predicted WHO-3'}, ...
            {'True WHO-1', 'True WHO-3'}, cm, ...
            'Colormap', parula, 'ColorbarVisible', 'on');
h.Title = sprintf('Test Set Confusion Matrix (n=%d)', size(X_test_pca, 1));
h.XLabel = 'Predicted Class';
h.YLabel = 'True Class';
saveas(gcf, 'results/meningioma_ftir_pipeline/test_confusion_matrix.png');

% --- ROC Curve ---
figure('Position', [100, 100, 700, 600]);
plot(X_roc, Y_roc, 'b-', 'LineWidth', 2);
hold on;
plot([0, 1], [0, 1], 'k--', 'LineWidth', 1);  % Chance line
xlabel('False Positive Rate (1-Specificity)');
ylabel('True Positive Rate (Sensitivity)');
title(sprintf('ROC Curve (AUC = %.3f)', AUC));
legend(sprintf('%s (AUC=%.3f)', final_model_package.classifier_type, AUC), ...
       'Chance', 'Location', 'southeast');
grid on;
saveas(gcf, 'results/meningioma_ftir_pipeline/test_roc_curve.png');

% --- PCA Scatter Plot with Predictions ---
figure('Position', [100, 100, 900, 600]);

% Correct predictions
correct_idx = (y_pred == testData.y);
scatter(X_test_pca(correct_idx & testData.y=='WHO-1', 1), ...
        X_test_pca(correct_idx & testData.y=='WHO-1', 2), ...
        100, 'b', 'filled', 'MarkerFaceAlpha', 0.7);
hold on;
scatter(X_test_pca(correct_idx & testData.y=='WHO-3', 1), ...
        X_test_pca(correct_idx & testData.y=='WHO-3', 2), ...
        100, 'r', 'filled', 'MarkerFaceAlpha', 0.7);

% Incorrect predictions (marked with X)
incorrect_idx = ~correct_idx;
scatter(X_test_pca(incorrect_idx & testData.y=='WHO-1', 1), ...
        X_test_pca(incorrect_idx & testData.y=='WHO-1', 2), ...
        150, 'b', 'x', 'LineWidth', 3);
scatter(X_test_pca(incorrect_idx & testData.y=='WHO-3', 1), ...
        X_test_pca(incorrect_idx & testData.y=='WHO-3', 2), ...
        150, 'r', 'x', 'LineWidth', 3);

xlabel('PC1'); ylabel('PC2');
legend({'WHO-1 (Correct)', 'WHO-3 (Correct)', ...
        'WHO-1 (Misclassified)', 'WHO-3 (Misclassified)'}, ...
        'Location', 'best');
title('Test Set Predictions in PCA Space');
grid on;
saveas(gcf, 'results/meningioma_ftir_pipeline/test_pca_predictions.png');
```

6. **Save All Results**
```matlab
% Create comprehensive results structure
test_results = struct();
test_results.predictions = y_pred;
test_results.scores = scores;
test_results.true_labels = testData.y;
test_results.probe_ids = testData.probe_ids;

% Performance metrics
test_results.metrics.accuracy = accuracy;
test_results.metrics.balanced_accuracy = balanced_accuracy;
test_results.metrics.sensitivity_WHO3 = sensitivity_WHO3;
test_results.metrics.specificity_WHO1 = specificity_WHO1;
test_results.metrics.ppv = ppv;
test_results.metrics.npv = npv;
test_results.metrics.f1 = f1;
test_results.metrics.f2 = f2;
test_results.metrics.auc = AUC;
test_results.metrics.confusion_matrix = cm;

% Misclassified samples (for review)
misclassified_idx = find(y_pred ~= testData.y);
test_results.misclassified_samples = testData.probe_ids(misclassified_idx);

save('results/Phase3_v2/test_results.mat', 'test_results');

% Create summary table
summary_table = table();
summary_table.Metric = {'Accuracy'; 'Balanced_Accuracy'; 'Sensitivity_WHO3'; ...
                        'Specificity_WHO1'; 'PPV'; 'NPV'; 'F1_Score'; ...
                        'F2_Score'; 'AUC_ROC'};
summary_table.Value = [accuracy; balanced_accuracy; sensitivity_WHO3; ...
                       specificity_WHO1; ppv; npv; f1; f2; AUC];
writetable(summary_table, 'results/Phase3_v2/test_performance.csv');
```

**Outputs**:
- `results/meningioma_ftir_pipeline/test_results.mat`: Complete results structure
- `results/meningioma_ftir_pipeline/test_performance.csv`: Metrics summary table
- `results/meningioma_ftir_pipeline/test_confusion_matrix.png`: Confusion matrix heatmap
- `results/meningioma_ftir_pipeline/test_roc_curve.png`: ROC curve
- `results/meningioma_ftir_pipeline/test_pca_predictions.png`: PCA scatter with predictions

---

## **PHASE 6: COMPREHENSIVE REPORTING & DOCUMENTATION**

### **Script: `generate_report.m`**

**Create a publication-ready report with**:

### **1. Executive Summary**

```markdown
## Performance Summary

**Training Set Cross-Validation (n=XX samples, 50 repeats × 5-fold CV):**
- Mean Balanced Accuracy: XX.X% ± X.X% (95% CI: XX.X-XX.X%)
- Mean Sensitivity (WHO-3): XX.X% ± X.X%
- Mean Specificity (WHO-1): XX.X% ± X.X%
- Mean F2-score (WHO-3): 0.XXX ± 0.XXX
- Mean AUC-ROC: 0.XXX ± 0.XXX

**Test Set Performance (n=XX samples, one-time evaluation):**
- Balanced Accuracy: XX.X%
- Sensitivity (WHO-3): XX.X%
- Specificity (WHO-1): XX.X%
- F2-score (WHO-3): 0.XXX
- AUC-ROC: 0.XXX

**Train-Test Gap Analysis:**
- Balanced Accuracy: -X.X% (indicates acceptable generalization)
- F2-score: -0.0XX
```

### **2. Detailed Methods Section (Manuscript-Ready)**

```markdown
## Methods

### Dataset and Quality Control

The dataset comprised 76 meningioma samples (38 WHO Grade 1, 38 WHO Grade 3) 
acquired via Fourier-transform infrared (FT-IR) microspectroscopy. Each sample 
yielded approximately 768 individual spectra. The dataset was divided into 
training (n=44, balanced 22:22 WHO-1:WHO-3) and test (n=32, balanced 16:16) 
sets prior to analysis. The test set was held out entirely during model 
development to ensure unbiased performance estimation.

Rigorous quality control was applied at multiple levels:

**Spectrum-level QC:** Individual spectra were evaluated for signal-to-noise 
ratio (SNR > 10), detector saturation (max absorbance < 1.8), baseline quality 
(SD < 0.02), and presence of characteristic biological peaks (Amide I/II ratio 
1.2-3.5). Spectra failing any criterion were excluded. Within-sample outliers 
were identified via Mahalanobis distance in principal component (PC) space 
(chi-squared threshold, 99% confidence).

**Sample-level QC:** Samples with mean within-sample spectral correlation < 0.85 
or fewer than 100 valid spectra after filtering were flagged for exclusion. 
Representative spectra were calculated as the quality-filtered mean of remaining 
spectra per sample (mean retention: XXX ± XX spectra per sample, range: XXX-XXX).

**Cross-sample QC:** Outlier samples were identified by calculating Mahalanobis 
distance separately within each WHO grade group to prevent false positives due 
to biological class differences. At most one sample per grade was excluded using 
a conservative 99% confidence threshold.

After QC, the final dataset comprised XX training samples (XX WHO-1, XX WHO-3) 
and XX test samples (XX WHO-1, XX WHO-3).

### Feature Extraction and Dimensionality Reduction

Principal component analysis (PCA) was performed exclusively on training set 
representative spectra to avoid data leakage. The number of components was 
selected to capture XX% of total variance (n=XX components). The resulting PCA 
transformation (loadings and mean) was then applied to test set spectra without 
re-computation.

### Model Development and Selection

Four classification algorithms were evaluated: Linear Discriminant Analysis (LDA), 
Partial Least Squares Discriminant Analysis (PLS-DA), Support Vector Machine 
(SVM) with radial basis function kernel, and Random Forest. Hyperparameter 
optimization was performed within stratified 5-fold cross-validation repeated 
50 times on the training set only. Performance was assessed using balanced 
accuracy, sensitivity for WHO Grade 3 detection, specificity for WHO Grade 1, 
F1-score, F2-score (emphasizing sensitivity), and area under the receiver 
operating characteristic curve (AUC-ROC).

The classifier achieving the highest mean F2-score ([CLASSIFIER_NAME], F2=0.XXX ± 0.XXX) 
was selected. This final model was trained on all XX training samples using 
optimal hyperparameters and evaluated once on the held-out test set.

### Statistical Analysis

Performance metrics are reported as mean ± standard deviation for cross-validation 
results and as point estimates for test set results. The 95% confidence intervals 
for CV metrics were calculated using the 2.5th and 97.5th percentiles across 
50 repeats. Train-test generalization gap was assessed by comparing CV mean 
performance to test set performance.
```

### **3. Results Tables**

**Table 1: Cross-Validation Performance (Training Set)**

```
| Classifier     | Bal. Acc (%) | Sensitivity (%) | Specificity (%) | F2-Score | AUC   |
|----------------|--------------|-----------------|-----------------|----------|-------|
| LDA            | XX.X ± X.X   | XX.X ± X.X      | XX.X ± X.X      | 0.XX±0.XX| 0.XX±0.XX |
| PLS-DA         | XX.X ± X.X   | XX.X ± X.X      | XX.X ± X.X      | 0.XX±0.XX| 0.XX±0.XX |
| SVM (RBF)      | XX.X ± X.X   | XX.X ± X.X      | XX.X ± X.X      | 0.XX±0.XX| 0.XX±0.XX |
| Random Forest  | XX.X ± X.X   | XX.X ± X.X      | XX.X ± X.X      | 0.XX±0.XX| 0.XX±0.XX |
```

**Table 2: Train vs. Test Performance (Best Model: [CLASSIFIER_NAME])**

```
| Metric                | Training CV (n=XX) | Test Set (n=XX) | Difference | Interpretation |
|-----------------------|--------------------|-----------------|------------|----------------|
| Balanced Accuracy (%) | XX.X ± X.X         | XX.X            | -X.X%      | Good           |
| Sensitivity WHO-3 (%) | XX.X ± X.X         | XX.X            | -X.X%      | Acceptable     |
| Specificity WHO-1 (%) | XX.X ± X.X         | XX.X            | -X.X%      | Good           |
| PPV (%)               | XX.X ± X.X         | XX.X            | -X.X%      | Good           |
| NPV (%)               | XX.X ± X.X         | XX.X            | -X.X%      | Good           |
| F1-Score              | 0.XXX ± 0.XXX      | 0.XXX           | -0.0XX     | Good           |
| F2-Score              | 0.XXX ± 0.XXX      | 0.XXX           | -0.0XX     | Good           |
| AUC-ROC               | 0.XXX ± 0.XXX      | 0.XXX           | -0.0XX     | Good           |
```

**Interpretation Guide:**
- Difference < 5%: Excellent generalization
- Difference 5-10%: Good generalization
- Difference 10-15%: Acceptable generalization
- Difference > 15%: Potential overfitting, interpret with caution

### **4. Quality Control Summary**

```markdown
## Quality Control Results

### Spectrum-Level Filtering
- Total spectra analyzed: ~XX,XXX (XX samples × ~768 spectra)
- Excluded for low SNR (< 10): X.X%
- Excluded for saturation (> 1.8 abs): X.X%
- Excluded for poor baseline (SD > 0.02): X.X%
- Excluded for anomalous peaks: X.X%
- Excluded as within-sample outliers: X.X%
- **Final retention rate: XX.X% (mean XXX ± XX spectra per sample)**

### Sample-Level Assessment
- Samples flagged for low within-sample correlation (< 0.85): X
- Samples flagged for insufficient valid spectra (< 100): X
- Samples identified as cross-sample outliers (Mahalanobis): X
- **Samples excluded from analysis: X (training), X (test)**

### Final Dataset
- Training: XX samples (XX WHO-1, XX WHO-3)
- Test: XX samples (XX WHO-1, XX WHO-3)
- Mean spectra per sample after QC: XXX ± XX
```

### **5. Discussion Points**

Generate interpretation text addressing:

1. **Model Performance**:
   - How does performance compare to literature?
   - Are sensitivity/specificity balanced appropriately for clinical use?
   - Is F2-score adequate for prioritizing high-grade detection?

2. **Generalization Assessment**:
   - Is train-test gap acceptable (target: < 10%)?
   - What does gap magnitude suggest about model robustness?

3. **Clinical Relevance**:
   - What is the clinical impact of misclassifications?
   - False negatives (missed WHO-3): Most concerning
   - False positives (over-diagnosed WHO-3): Less critical but increases unnecessary interventions

4. **Limitations**:
   - Small sample size (n=76 total)
   - Single-center data (potential site-specific bias)
   - Representative spectrum approach (loses intra-sample heterogeneity information)
   - Class balance may not reflect real-world prevalence

5. **Strengths**:
   - Rigorous QC at multiple levels
   - Strict train/test separation with no leakage
   - Conservative outlier detection to preserve samples
   - Multiple performance metrics reported
   - Transparent methodology

### **6. Save Complete Report**

```matlab
% Generate PDF report using MATLAB Report Generator or export to Word
% Include all tables, figures, and text sections

% Save final report
% Either as .docx (editable) or .pdf (publication-ready)
```

**Outputs**:
- `results/meningioma_ftir_pipeline/final_report.pdf` or `.docx`
- `results/meningioma_ftir_pipeline/methods_section.txt`: Copy-paste ready methods text
- `results/meningioma_ftir_pipeline/results_tables.xlsx`: All tables in spreadsheet format

---

## **CODE QUALITY & REPRODUCIBILITY REQUIREMENTS**

### **1. Modularity**

Each phase should be a separate, callable function/script:

```matlab
% Main pipeline script
function run_full_pipeline()
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
end
```

### **2. Error Handling**

```matlab
% Check for missing files
if ~exist('data/data_table_train.mat', 'file')
    error('Training data file not found. Check data/ directory.');
end

% Verify dimensions
assert(size(X_train, 2) == size(X_test, 2), ...
       'Training and test sets have different numbers of wavenumbers');

% Check for NaN/Inf
if any(isnan(X_train(:)))
    warning('Training data contains NaN values. Review QC process.');
end
```

### **3. Logging**

```matlab
% Create log file with timestamps
log_file = fopen('results/meningioma_ftir_pipeline/pipeline_log.txt', 'w');

function log_message(msg)
    timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    fprintf(log_file, '[%s] %s\n', timestamp, msg);
    fprintf('[%s] %s\n', timestamp, msg);  % Also print to console
end

% Usage
log_message('Starting Phase 0: Quality Control');
log_message(sprintf('Loaded %d training samples', size(X_train, 1)));
```

### **4. Reproducibility**

```matlab
% Set random seed at the start of EVERY script that uses randomness
rng(42, 'twister');  % Use Mersenne Twister algorithm

% Document MATLAB version and toolboxes
ver  % Print version information
```

### **5. Configuration File**

Create `config_phase3_v2.m`:

```matlab
function cfg = config_phase3_v2()
    % Configuration for Meningioma FT-IR Classification Pipeline
    % Phase 3 Version 2 with Integrated QC
    
    cfg = struct();
    
    % === PATHS ===
    cfg.paths.data = 'data/';
    cfg.paths.models = 'models/Phase3_v2/';
    cfg.paths.results = 'results/Phase3_v2/';
    cfg.paths.qc = 'results/Phase3_v2/qc/';
    
    % === QUALITY CONTROL ===
    cfg.qc.snr_threshold = 10;
    cfg.qc.max_absorbance = 1.8;
    cfg.qc.baseline_sd_threshold = 0.02;
    cfg.qc.min_absorbance = -0.1;
    cfg.qc.amide_ratio_min = 1.2;
    cfg.qc.amide_ratio_max = 3.5;
    cfg.qc.within_sample_corr_threshold = 0.85;
    cfg.qc.min_spectra_per_sample = 100;
    cfg.qc.outlier_confidence = 0.99;  % Chi-squared threshold
    cfg.qc.max_samples_to_exclude = 2;  % Conservative limit
    
    % === PCA ===
    cfg.pca.variance_threshold = 0.95;  % Keep PCs explaining 95% variance
    cfg.pca.max_components = 15;  % Upper limit on components
    
    % === CROSS-VALIDATION ===
    cfg.cv.n_folds = 5;
    cfg.cv.n_repeats = 50;
    cfg.cv.stratified = true;
    
    % === CLASSIFIERS ===
    cfg.classifiers.types = {'LDA', 'PLSDA', 'SVM', 'RandomForest'};
    
    % Hyperparameter grids
    cfg.classifiers.plsda_n_components = 1:10;
    cfg.classifiers.svm_C_values = 10.^(-2:0.5:2);
    cfg.classifiers.svm_gamma_values = 10.^(-3:0.5:1);
    cfg.classifiers.rf_n_trees = [50, 100, 200, 500];
    cfg.classifiers.rf_max_depth = [5, 10, 20, 30];
    
    % === REPRODUCIBILITY ===
    cfg.random_seed = 42;
    
    % === REPORTING ===
    cfg.report.figures_format = 'png';
    cfg.report.figures_dpi = 300;
    cfg.report.save_intermediate = true;
end
```

### **6. Documentation**

Each function must have clear header documentation:

```matlab
function representative_spectrum = calculate_qc_filtered_mean(spectra, wavenumbers, cfg)
% CALCULATE_QC_FILTERED_MEAN - Compute quality-filtered mean representative spectrum
%
% Syntax:
%   representative_spectrum = calculate_qc_filtered_mean(spectra, wavenumbers, cfg)
%
% Inputs:
%   spectra      - Matrix of individual spectra [n_spectra × n_wavenumbers]
%   wavenumbers  - Vector of wavenumber values [1 × n_wavenumbers]
%   cfg          - Configuration structure with QC thresholds
%
% Outputs:
%   representative_spectrum - Mean of quality-filtered spectra [1 × n_wavenumbers]
%
% Description:
%   Applies multi-level quality control filtering:
%   1. SNR filtering (threshold: cfg.qc.snr_threshold)
%   2. Saturation check (threshold: cfg.qc.max_absorbance)
%   3. Baseline quality (threshold: cfg.qc.baseline_sd_threshold)
%   4. Amide peak ratio check (range: cfg.qc.amide_ratio_min to max)
%   5. Mahalanobis distance outlier removal (confidence: cfg.qc.outlier_confidence)
%
% Example:
%   cfg = config_phase3_v2();
%   rep_spec = calculate_qc_filtered_mean(sample_spectra, wavenumbers, cfg);
%
% See also: QUALITY_CONTROL_ANALYSIS, MAHAL, PCA

% Author: [Your Name]
% Date: [Date]
% Version: 1.0

    % [Function implementation here]
end
```

---

## **VALIDATION CHECKLIST**

Before delivering code, verify:

**Data Integrity:**
- [ ] Training and test sets never mixed during QC or preprocessing
- [ ] QC parameters applied consistently to both sets
- [ ] No data-driven QC parameters leak from test to training

**PCA Workflow:**
- [ ] PCA computed only on training set representative spectra
- [ ] Training mean (mu) saved and used for test set centering
- [ ] Training loadings (coeff) used for test set projection
- [ ] No re-computation of PCA on test set

**Cross-Validation:**
- [ ] CV performed only on training set
- [ ] Stratification ensures balanced folds
- [ ] CV repeated 50 times for robust estimates
- [ ] Hyperparameter optimization within CV, not on test set

**Test Set Protocol:**
- [ ] Test set evaluated exactly once
- [ ] No iterative adjustments based on test performance
- [ ] All metrics calculated and reported (not cherry-picked)

**Quality Control:**
- [ ] Spectrum-level filters applied with documented thresholds
- [ ] Sample-level consistency checks performed
- [ ] Outlier detection uses conservative thresholds
- [ ] QC decisions documented and justified

**Reproducibility:**
- [ ] Random seeds set in all scripts (rng(42))
- [ ] Configuration file contains all parameters
- [ ] Code runs end-to-end without errors
- [ ] All intermediate results saved

**Outputs:**
- [ ] All required files generated in correct folders
- [ ] Figures publication-quality (300 DPI, clear labels)
- [ ] Tables formatted for manuscript inclusion
- [ ] Report comprehensive and accurate

**Scientific Rigor:**
- [ ] Train-test performance gap calculated and interpreted
- [ ] Confidence intervals reported for CV results
- [ ] Limitations acknowledged
- [ ] Methods section manuscript-ready

---

## **DELIVERABLES SUMMARY**

### **1. Code** (`src/phase3_v2/`)
- `config_phase3_v2.m`: Configuration file
- `quality_control_analysis.m`: Phase 0
- `load_and_prepare_data.m`: Phase 1
- `perform_feature_selection.m`: Phase 2
- `run_cross_validation.m`: Phase 3
- `train_final_model.m`: Phase 4
- `evaluate_test_set.m`: Phase 5
- `generate_report.m`: Phase 6
- `run_full_pipeline.m`: Master script
- `README.md`: Usage instructions

### **2. Models** (`models/Phase3_v2/`)
- `pca_model.mat`: PCA transformation parameters
- `final_model.mat`: Trained classifier with metadata

### **3. Results** (`results/Phase3_v2/`)

**QC Outputs** (`qc/` subfolder):
- `qc_report.pdf`: Comprehensive QC documentation
- `qc_metrics_summary.csv`: Per-sample QC statistics
- `cleaned_train_representatives.mat`: QC'd training spectra
- `cleaned_test_representatives.mat`: QC'd test spectra
- `qc_*.png`: QC visualization figures (SNR distribution, correlations, PCA outliers, etc.)

**Modeling Outputs**:
- `preprocessed_data.mat`: Analysis-ready datasets
- `X_train_pca.mat`: Transformed training data
- `pca_variance_explained.png`: Variance plots
- `cv_performance.csv`: CV results table
- `best_classifier_selection.mat`: Selected model info
- `test_results.mat`: Complete test set results
- `test_performance.csv`: Test metrics summary
- `test_confusion_matrix.png`
- `test_roc_curve.png`
- `test_pca_predictions.png`

**Report**:
- `final_report.pdf` or `.docx`: Publication-ready comprehensive report
- `methods_section.txt`: Copy-paste ready methods text
- `results_tables.xlsx`: All tables formatted for manuscript
- `pipeline_log.txt`: Complete execution log

---

## **FINAL CONFIRMATION QUESTIONS**

**Please confirm your understanding of:**

1. **QC Philosophy**: 
   - QC ensures data quality using fixed thresholds or training-set-derived parameters
   - QC is distinct from modeling and does not constitute data leakage
   - Representative spectrum calculation (quality-filtered mean) happens before modeling

2. **Train/Test Separation**:
   - Training set (after QC): ~42-44 samples for all development
   - Test set (after QC): ~30-32 samples for one-time evaluation
   - No parameters learned from test set at any stage

3. **CV Strategy**:
   - Simple stratified K-fold on training set only
   - Separate test set makes nested CV unnecessary
   - CV for hyperparameter tuning and model selection

4. **PCA Workflow**:
   - Compute PCA on training representative spectra
   - Apply transformation to test set using training parameters
   - No re-fitting on test data

5. **One-Time Test Evaluation**:
   - Test set touched exactly once
   - All metrics reported transparently
   - Performance gap calculated and interpreted

6. **Expected Outputs**:
   - Comprehensive QC documentation
   - CV performance with confidence intervals
   - Test performance with visualizations
   - Publication-ready methods and results sections

**If you have ANY questions or uncertainties about any aspect of this integrated workflow, ask now before you start coding. This ensures a scientifically valid and reproducible pipeline.**

---

## **APPENDIX: HELPER FUNCTIONS**

Suggested utility functions to implement:

```matlab
% Calculate all classification metrics from predictions
function metrics = calculate_classification_metrics(y_true, y_pred, scores)

% Generate publication-quality confusion matrix figure
function fig = plot_confusion_matrix(cm, class_names, title_str)

% Create ROC curve with confidence intervals
function fig = plot_roc_curve_with_ci(y_true, scores, n_bootstrap)

% Apply Mahalanobis distance outlier detection
function outlier_mask = detect_mahalanobis_outliers(data, confidence)

% Calculate within-sample spectral correlation
function mean_corr = calculate_within_sample_correlation(spectra)

% Log progress with timestamps
function log_message(message, log_file_handle)
```

---

**This integrated workflow ensures:**
✓ Rigorous multi-level quality control  
✓ No data leakage between train and test sets  
✓ Conservative outlier detection preserving sample size  
✓ Transparent methodology with full documentation  
✓ Publication-ready outputs and manuscript sections  
✓ Reproducible results with fixed random seeds  
✓ Scientifically sound machine learning pipeline  

**You are now ready to implement this pipeline. Good luck!**
