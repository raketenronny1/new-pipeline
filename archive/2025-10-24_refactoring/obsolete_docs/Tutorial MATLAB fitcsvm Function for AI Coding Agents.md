

## Overview

`fitcsvm` trains Support Vector Machine (SVM) classifiers for **one-class** and **two-class (binary) classification** on low-to-moderate dimensional data.

---

## Core Capabilities

### Function Purpose

- **Binary Classification**: Separates data into two classes
- **One-Class Learning**: Detects outliers/anomalies (all labels are same class)
- **Kernel Methods**: Supports linear, Gaussian (RBF), and polynomial kernels
- **Hyperparameter Optimization**: Automatic tuning available

### When to Use fitcsvm

✅ **Use when:**

- You have 2 classes or need anomaly detection
- Feature space is low-to-moderate dimensional (< 10,000 predictors)
- You need interpretable support vectors
- Non-linear decision boundaries are needed

❌ **Don't use when:**

- You have > 2 classes (use `fitcecoc` instead)
- High-dimensional data (use `fitclinear` instead)
- Very large datasets (> 100,000 observations with non-linear kernels)

---

## Basic Syntax Patterns

### Pattern 1: Simple Binary Classification

```matlab
% Load data
load fisheriris
X = meas(51:end, 3:4);  % Use 2 features, exclude setosa
Y = species(51:end);     % Binary: versicolor vs virginica

% Train SVM
Mdl = fitcsvm(X, Y);

% Predict
[label, score] = predict(Mdl, X);
```

### Pattern 2: With Table Input

```matlab
% Data in table format
Tbl = readtable('data.csv');

% Specify response variable name
Mdl = fitcsvm(Tbl, 'ResponseVariable');

% Or use formula
Mdl = fitcsvm(Tbl, 'Response ~ Predictor1 + Predictor2');
```

### Pattern 3: One-Class Learning (Anomaly Detection)

```matlab
% All labels are the same
X = meas(:, 1:2);
Y = ones(size(X, 1), 1);  % Single class

% Train with outlier fraction
Mdl = fitcsvm(X, Y, 'OutlierFraction', 0.05);

% Negative scores indicate anomalies
[~, scores] = predict(Mdl, X);
isAnomaly = scores < 0;
```

---

## Critical Parameters for Best Practices

### 1. **Standardize** (MOST IMPORTANT)

```matlab
% ALWAYS standardize unless you have specific reasons not to
Mdl = fitcsvm(X, Y, 'Standardize', true);
```

**Why:** Makes predictors insensitive to scale, improves optimization

### 2. **KernelFunction**

```matlab
% Linear (default for 2-class, fastest)
Mdl = fitcsvm(X, Y, 'KernelFunction', 'linear');

% Gaussian/RBF (default for 1-class, handles non-linearity)
Mdl = fitcsvm(X, Y, 'KernelFunction', 'rbf');

% Polynomial
Mdl = fitcsvm(X, Y, 'KernelFunction', 'polynomial', 'PolynomialOrder', 3);
```

### 3. **BoxConstraint** (Regularization)

```matlab
% Higher = more complex model (less regularization)
% Lower = simpler model (more regularization)
Mdl = fitcsvm(X, Y, 'BoxConstraint', 1);  % Default

% For noisy data, reduce:
Mdl = fitcsvm(X, Y, 'BoxConstraint', 0.1);
```

### 4. **KernelScale** (For Non-Linear Kernels)

```matlab
% Auto-select (RECOMMENDED)
Mdl = fitcsvm(X, Y, 'KernelFunction', 'rbf', 'KernelScale', 'auto');

% Manual selection
Mdl = fitcsvm(X, Y, 'KernelFunction', 'rbf', 'KernelScale', 2.5);
```

---

## Best Practice Workflow

### Step-by-Step Implementation

```matlab
%% 1. LOAD AND PREPARE DATA
load fisheriris
X = meas(51:end, :);
Y = species(51:end);

%% 2. STANDARDIZE (Critical for SVM)
% fitcsvm will handle this internally
% Do NOT manually standardize before calling fitcsvm

%% 3. SPECIFY CLASS ORDER (Good Practice)
Mdl = fitcsvm(X, Y, ...
    'ClassNames', {'versicolor', 'virginica'}, ...  % Explicit order
    'Standardize', true);

%% 4. CROSS-VALIDATE
CVMdl = crossval(Mdl, 'KFold', 10);
loss = kfoldLoss(CVMdl);
fprintf('Cross-validation error: %.2f%%\n', loss * 100);

%% 5. PREDICT ON NEW DATA
newData = [5.0, 2.0, 3.5, 1.0];
[label, score] = predict(Mdl, newData);
```

---

## Advanced: Hyperparameter Optimization

### Automatic Tuning (RECOMMENDED)

```matlab
% Let MATLAB find optimal parameters
Mdl = fitcsvm(X, Y, ...
    'OptimizeHyperparameters', 'auto', ...  % Optimize BoxConstraint, KernelScale, Standardize
    'HyperparameterOptimizationOptions', ...
    struct('AcquisitionFunctionName', 'expected-improvement-plus', ...
           'MaxObjectiveEvaluations', 30, ...
           'ShowPlots', false, ...
           'Verbose', 0));
```

### Custom Optimization

```matlab
% Specify which parameters to optimize
Mdl = fitcsvm(X, Y, ...
    'OptimizeHyperparameters', {'BoxConstraint', 'KernelScale'}, ...
    'HyperparameterOptimizationOptions', ...
    struct('MaxObjectiveEvaluations', 50, ...
           'KFold', 5));
```

---

## Handling Class Imbalance

```matlab
% Option 1: Adjust prior probabilities
Mdl = fitcsvm(X, Y, ...
    'Prior', 'uniform');  % Equal weight to both classes

% Option 2: Adjust misclassification costs
costMatrix = [0, 2; 1, 0];  % Penalize false negatives more
Mdl = fitcsvm(X, Y, 'Cost', costMatrix);

% Option 3: Observation weights
weights = ones(size(Y));
weights(Y == 'minorityClass') = 2;  % Double weight for minority
Mdl = fitcsvm(X, Y, 'Weights', weights);
```

---

## Performance Optimization Tips

### 1. For Large Datasets

```matlab
% Use larger cache
Mdl = fitcsvm(X, Y, 'CacheSize', 'maximal');

% Remove duplicates (speeds up convergence)
Mdl = fitcsvm(X, Y, 'RemoveDuplicates', true);

% Enable shrinkage
Mdl = fitcsvm(X, Y, 'ShrinkagePeriod', 1000);
```

### 2. For Faster Training

```matlab
% Use linear kernel for linearly separable data
Mdl = fitcsvm(X, Y, 'KernelFunction', 'linear');

% Increase box constraint (fewer support vectors)
Mdl = fitcsvm(X, Y, 'BoxConstraint', 10);

% Set iteration limit
Mdl = fitcsvm(X, Y, 'IterationLimit', 1e5);
```

---

## Common Patterns and Checks

### Pattern: Check Model Quality

```matlab
% Training data performance
resubLoss = resubLoss(Mdl);

% Cross-validation
CVMdl = crossval(Mdl);
cvLoss = kfoldLoss(CVMdl);

% Number of support vectors (fewer is better)
numSV = sum(Mdl.IsSupportVector);
fprintf('Support vectors: %d / %d (%.1f%%)\n', ...
    numSV, length(Mdl.IsSupportVector), 100*numSV/length(Mdl.IsSupportVector));
```

### Pattern: Visualize Decision Boundary (2D)

```matlab
% For 2 predictors only
[x1Grid, x2Grid] = meshgrid(linspace(min(X(:,1)), max(X(:,1)), 100), ...
                             linspace(min(X(:,2)), max(X(:,2)), 100));
[~, scores] = predict(Mdl, [x1Grid(:), x2Grid(:)]);
scores = reshape(scores(:,2), size(x1Grid));

contourf(x1Grid, x2Grid, scores);
hold on;
gscatter(X(:,1), X(:,2), Y);
```

---

## Error Handling and Validation

```matlab
% Check for valid inputs
assert(size(X, 1) == length(Y), 'X and Y must have same number of rows');
assert(~any(isnan(X(:))), 'X contains NaN values');
assert(~any(isinf(X(:))), 'X contains Inf values');

% Check class distribution
classCounts = countcats(categorical(Y));
if length(classCounts) > 2
    error('fitcsvm only supports binary classification. Use fitcecoc for multiclass.');
end

% Warn for imbalanced data
if min(classCounts) / max(classCounts) < 0.3
    warning('Class imbalance detected. Consider adjusting Prior or Cost.');
end
```

---

## Complete Example: Production-Ready Code

```matlab
function Mdl = trainSVMClassifier(X, Y, options)
    % TRAINSVMCLASSIFIER Train SVM with best practices
    %
    % Inputs:
    %   X - Predictors (n x p matrix)
    %   Y - Labels (n x 1 vector)
    %   options - struct with optional fields:
    %       .optimize (logical): Run hyperparameter optimization
    %       .crossValidate (logical): Perform cross-validation
    %       .verbose (logical): Display progress
    
    arguments
        X double
        Y
        options.optimize (1,1) logical = false
        options.crossValidate (1,1) logical = true
        options.verbose (1,1) logical = true
    end
    
    % Validate inputs
    assert(size(X, 1) == length(Y), 'Dimension mismatch');
    
    % Check for one-class vs two-class
    uniqueClasses = unique(Y);
    isOneClass = length(uniqueClasses) == 1;
    
    if options.verbose
        fprintf('Training %s SVM...\n', ...
            ternary(isOneClass, 'one-class', 'binary'));
    end
    
    % Set up base parameters
    baseArgs = {'Standardize', true};
    
    if isOneClass
        baseArgs = [baseArgs, {'OutlierFraction', 0.05}];
    end
    
    % Hyperparameter optimization
    if options.optimize && ~isOneClass
        baseArgs = [baseArgs, ...
            {'OptimizeHyperparameters', 'auto', ...
             'HyperparameterOptimizationOptions', ...
             struct('MaxObjectiveEvaluations', 30, ...
                    'ShowPlots', false, ...
                    'Verbose', options.verbose)}];
    end
    
    % Train model
    Mdl = fitcsvm(X, Y, baseArgs{:});
    
    % Cross-validation
    if options.crossValidate && ~isOneClass
        CVMdl = crossval(Mdl);
        cvLoss = kfoldLoss(CVMdl);
        
        if options.verbose
            fprintf('Cross-validation loss: %.4f\n', cvLoss);
            fprintf('Support vectors: %d / %d\n', ...
                sum(Mdl.IsSupportVector), length(Mdl.IsSupportVector));
        end
    end
end

function result = ternary(condition, trueVal, falseVal)
    if condition
        result = trueVal;
    else
        result = falseVal;
    end
end
```

---

## Summary Checklist for AI Agent

✅ **Always do:**

1. Set `'Standardize', true`
2. Validate binary classification (2 classes or 1 class for anomaly detection)
3. Use cross-validation to assess generalization
4. Check for class imbalance and address if needed
5. Start with `'KernelScale', 'auto'` for non-linear kernels

✅ **Consider:**

1. Hyperparameter optimization for best performance
2. Linear kernel for linearly separable data (faster)
3. Removing duplicates for large datasets
4. Adjusting BoxConstraint based on data noise

❌ **Avoid:**

1. Using fitcsvm for > 2 classes (use fitcecoc)
2. High-dimensional data without feature selection (use fitclinear)
3. Manually standardizing data before calling fitcsvm
4. Ignoring convergence warnings

---

## Quick Reference: Key Parameters

|Parameter|Purpose|Typical Values|
|---|---|---|
|`Standardize`|Scale features|**true** (always)|
|`KernelFunction`|Boundary type|'linear', 'rbf', 'polynomial'|
|`BoxConstraint`|Regularization|0.1-100 (1 default)|
|`KernelScale`|Kernel width|**'auto'** or positive scalar|
|`OutlierFraction`|Anomaly rate|0.01-0.1 for one-class|
|`OptimizeHyperparameters`|Auto-tune|**'auto'** recommended|