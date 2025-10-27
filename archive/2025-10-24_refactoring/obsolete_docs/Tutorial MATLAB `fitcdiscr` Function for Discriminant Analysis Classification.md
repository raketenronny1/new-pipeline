
## Overview

`fitcdiscr` fits a discriminant analysis classifier in MATLAB. It's used for supervised classification problems where you want to predict class labels based on predictor variables.

## Basic Syntax

matlab

```matlab
% Most common usage patterns:
Mdl = fitcdiscr(X, Y)                        % Predictor matrix and response vector
Mdl = fitcdiscr(Tbl, ResponseVarName)        % Table with response variable name
Mdl = fitcdiscr(Tbl, formula)                % Table with formula
Mdl = fitcdiscr(___, Name=Value)             % With additional options
```

## Key Concepts

### Discriminant Types

Choose the appropriate discriminant type based on your assumptions about covariance:

- **'linear'** (default): Regularized LDA - all classes have same covariance matrix
- **'diaglinear'**: LDA with diagonal covariance (faster, assumes independent features)
- **'quadratic'**: QDA - covariance can vary among classes (more flexible)
- **'diagquadratic'**: QDA with diagonal covariance
- **'pseudolinear'**: LDA using pseudo-inverse (for singular matrices)
- **'pseudoquadratic'**: QDA using pseudo-inverse

### Model Assumption

The model assumes each class generates data using a multivariate normal distribution (Gaussian mixture).

## Essential Parameters

### 1. **DiscrimType** - Discriminant type

matlab

```matlab
Mdl = fitcdiscr(X, Y, 'DiscrimType', 'quadratic');
```

### 2. **Prior** - Prior probabilities

matlab

```matlab
% Options:
'empirical'  % Based on class frequencies (default)
'uniform'    % Equal probability for all classes
[0.3, 0.7]   % Custom probabilities (must sum to 1)
```

### 3. **Cost** - Misclassification cost matrix

matlab

```matlab
% Cost(i,j) = cost of classifying as j when true class is i
Cost = [0 1; 2 0];  % Higher cost for false negatives
Mdl = fitcdiscr(X, Y, 'Cost', Cost);
```

### 4. **Gamma** - Regularization (for linear discriminant only)

matlab

```matlab
% Range [0,1]: controls covariance matrix structure
Mdl = fitcdiscr(X, Y, 'Gamma', 0.5);  % 0=no regularization, 1=maximum
```

### 5. **Delta** - Linear coefficient threshold

matlab

```matlab
% Sets small coefficients to 0 for feature selection
Mdl = fitcdiscr(X, Y, 'Delta', 0.01);
```

## Cross-Validation

Always use cross-validation to assess model performance:

matlab

```matlab
% K-fold cross-validation
Mdl = fitcdiscr(X, Y, 'KFold', 5);

% Holdout validation
Mdl = fitcdiscr(X, Y, 'Holdout', 0.2);  % 20% test set

% Leave-one-out
Mdl = fitcdiscr(X, Y, 'Leaveout', 'on');

% Custom partition
cvp = cvpartition(Y, 'KFold', 10);
Mdl = fitcdiscr(X, Y, 'CVPartition', cvp);
```

## Hyperparameter Optimization

**CRITICAL**: This is the most powerful feature for automatic model tuning:

matlab

```matlab
% Automatic optimization
Mdl = fitcdiscr(X, Y, ...
    'OptimizeHyperparameters', 'auto', ...  % Optimizes Delta and Gamma
    'HyperparameterOptimizationOptions', ...
    struct('AcquisitionFunctionName', 'expected-improvement-plus', ...
           'MaxObjectiveEvaluations', 30));

% Optimize all eligible parameters
Mdl = fitcdiscr(X, Y, 'OptimizeHyperparameters', 'all');

% Optimize specific parameters
Mdl = fitcdiscr(X, Y, 'OptimizeHyperparameters', {'Delta', 'DiscrimType'});
```

**Eligible parameters for optimization:**

- `Delta`: Positive values (log-scaled [1e-6, 1e3])
- `DiscrimType`: All 6 types
- `Gamma`: Range [0, 1]

## Complete Workflow Example

matlab

```matlab
% 1. Load and prepare data
load fisheriris
X = meas;  % Predictors
Y = species;  % Response

% 2. Basic model
MdlBasic = fitcdiscr(X, Y);

% 3. Model with options
Mdl = fitcdiscr(X, Y, ...
    'DiscrimType', 'linear', ...
    'Prior', 'uniform', ...
    'KFold', 5);

% 4. Optimized model (RECOMMENDED)
rng(1);  % For reproducibility
MdlOpt = fitcdiscr(X, Y, ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', ...
    struct('AcquisitionFunctionName', 'expected-improvement-plus', ...
           'MaxObjectiveEvaluations', 30, ...
           'Verbose', 1, ...
           'ShowPlots', false));

% 5. Access model properties
MdlOpt.Mu  % Class means
MdlOpt.ClassNames  % Class labels

% 6. Make predictions (on new model without cross-validation)
MdlFinal = fitcdiscr(X, Y, 'DiscrimType', MdlOpt.DiscrimType);
[labels, scores] = predict(MdlFinal, X);
```

## Best Practices for AI Agents

### 1. **Always Handle Missing Data**

matlab

```matlab
% Remove missing values before training
R = rmmissing([X, Y]);
X = R(:, 1:end-1);
Y = R(:, end);
```

### 2. **Standardize Features** (especially for regularization)

matlab

```matlab
X_std = zscore(X);
Mdl = fitcdiscr(X_std, Y);
```

### 3. **Start with Optimization**

matlab

```matlab
% Let MATLAB find best hyperparameters first
Mdl = fitcdiscr(X, Y, 'OptimizeHyperparameters', 'auto');
```

### 4. **Use Cross-Validation for Model Selection**

matlab

```matlab
% Don't use the same data for training and testing
Mdl = fitcdiscr(X, Y, 'KFold', 5);
```

### 5. **Check Class Balance**

matlab

```matlab
% If imbalanced, adjust priors or costs
classCounts = histcounts(Y);
if max(classCounts)/min(classCounts) > 3
    Mdl = fitcdiscr(X, Y, 'Prior', 'uniform');  % Or use custom costs
end
```

### 6. **Memory Management for Large Datasets**

matlab

```matlab
% For many predictors
Mdl = fitcdiscr(X, Y, 'SaveMemory', 'on');
```

### 7. **Proper Output Format**

matlab

```matlab
% When cross-validating, Mdl is ClassificationPartitionedModel
% Access trained model: Mdl.Trained{1}
% Otherwise, Mdl is ClassificationDiscriminant
```

## Common Patterns

### Pattern 1: Quick Classification

matlab

```matlab
Mdl = fitcdiscr(X, Y);
predictions = predict(Mdl, X_test);
```

### Pattern 2: Optimized with Cross-Validation

matlab

```matlab
rng('default');
Mdl = fitcdiscr(X, Y, ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', ...
    struct('MaxObjectiveEvaluations', 30, ...
           'Holdout', 0.3));
```

### Pattern 3: Custom Cost Matrix (e.g., medical diagnosis)

matlab

```matlab
% False negatives more costly than false positives
Cost = [0 1; 5 0];  % Rows=true, Cols=predicted
Mdl = fitcdiscr(X, Y, 'Cost', Cost, 'KFold', 5);
```

### Pattern 4: Feature Selection with Delta

matlab

```matlab
Mdl = fitcdiscr(X, Y, 'Delta', 0.1);
% Check which features were kept: find(abs(coefficients) > 0)
```

## Important Notes

1. **Categorical predictors are NOT supported** - encode them numerically first
2. **Delta must be 0 for quadratic models**
3. **Gamma only works with linear discriminant types**
4. **Cost matrix**: Cost(i,j) = cost of predicting j when true class is i
5. **Cannot use cross-validation arguments with OptimizeHyperparameters** - use HyperparameterOptimizationOptions instead

## Output

The function returns:

- `ClassificationDiscriminant` object (regular model)
- `ClassificationPartitionedModel` object (cross-validated model)

Access properties with dot notation:

matlab

```matlab
Mdl.Mu                    % Class means
Mdl.Sigma                 % Covariance matrices
Mdl.ClassNames            % Class labels
Mdl.Prior                 % Prior probabilities
Mdl.Cost                  % Cost matrix
Mdl.DiscrimType           % Discriminant type used
```

## When to Use Discriminant Analysis

**Good for:**

- Linear or quadratic decision boundaries
- Gaussian-distributed features
- Multiclass classification
- When interpretability is important (linear models)
- Small to medium datasets

**Consider alternatives if:**

- Features are not approximately normal
- Need highly non-linear boundaries (use SVM, trees, neural networks)
- High-dimensional sparse data (use naive Bayes, SVM)