

## Overview

`fitcensemble` creates an ensemble of trained classification models by combining multiple weak learners. This function is part of MATLAB's Statistics and Machine Learning Toolbox and is used for supervised learning classification tasks.

**Primary Purpose**: Train ensemble classification models using various aggregation methods (boosting, bagging, random subspace) to improve predictive accuracy.

**Default Behavior**:

- Uses **LogitBoost** for binary classification
- Uses **AdaBoostM2** for multiclass classification
- Trains **100 classification trees** by default

---

## Function Syntax

### Basic Syntaxes

matlab

```matlab
% Using table with response variable name
Mdl = fitcensemble(Tbl, ResponseVarName)

% Using table with formula
Mdl = fitcensemble(Tbl, formula)

% Using table with separate response array
Mdl = fitcensemble(Tbl, Y)

% Using predictor matrix and response array
Mdl = fitcensemble(X, Y)

% With additional name-value arguments
Mdl = fitcensemble(___, Name, Value)

% With hyperparameter optimization results
[Mdl, AggregateOptimizationResults] = fitcensemble(___)
```

---

## Input Arguments

### Data Input Formats

#### 1. **Table Format (Tbl)**

matlab

```matlab
% Tbl: Table where each row = observation, each column = predictor
% - Can contain response variable or specify separately
% - Multicolumn variables and cell arrays (except character vectors) not allowed
```

#### 2. **Matrix Format (X, Y)**

matlab

```matlab
% X: Numeric matrix (n × p)
%    - n observations (rows)
%    - p predictors (columns)
% Y: Response labels (categorical, character, string, logical, numeric, or cell array)
```

#### 3. **Formula Format**

matlab

```matlab
% formula: 'Y ~ x1 + x2 + x3'
% Specifies response and predictor subset
```

---

## Key Parameters

### Ensemble Configuration

#### **Method** - Ensemble Aggregation Algorithm

|Value|Method|Problem Type|Related Parameters|
|---|---|---|---|
|`'Bag'`|Bootstrap aggregation (bagging/random forest)|Binary & Multiclass|N/A|
|`'Subspace'`|Random subspace|Binary & Multiclass|`NPredToSample`|
|`'AdaBoostM1'`|Adaptive boosting|Binary only|`LearnRate`|
|`'AdaBoostM2'`|Adaptive boosting|Multiclass only|`LearnRate`|
|`'GentleBoost'`|Gentle adaptive boosting|Binary only|`LearnRate`|
|`'LogitBoost'`|Adaptive logistic regression|Binary only|`LearnRate`|
|`'LPBoost'`|Linear programming boosting|Binary & Multiclass|`MarginPrecision`|
|`'RobustBoost'`|Robust boosting|Binary only|`RobustErrorGoal`, `RobustMarginSigma`|
|`'RUSBoost'`|Random undersampling boosting|Binary & Multiclass|`LearnRate`, `RatioToSmallest`|
|`'TotalBoost'`|Totally corrective boosting|Binary & Multiclass|`MarginPrecision`|

**Example**:

matlab

```matlab
Mdl = fitcensemble(X, Y, 'Method', 'Bag')
```

---

#### **NumLearningCycles** - Number of Learning Cycles

- **Default**: 100
- **Range**: Positive integer or `'AllPredictorCombinations'`
- **Effect**: Total learners = `NumLearningCycles × numel(Learners)`

**Example**:

matlab

```matlab
Mdl = fitcensemble(X, Y, 'NumLearningCycles', 500)
```

---

#### **Learners** - Weak Learner Types

|Weak Learner|Name|Template Function|Recommended Method|
|---|---|---|---|
|Decision Tree|`'tree'`|`templateTree`|All except 'Subspace'|
|Discriminant Analysis|`'discriminant'`|`templateDiscriminant`|'Subspace' recommended|
|k-Nearest Neighbors|`'knn'`|`templateKNN`|'Subspace' only|

**Default Tree Settings**:

- **Bagging**: `MaxNumSplits = n-1` (deep trees), `NumVariablesToSample = sqrt(p)`
- **Boosting**: `MaxNumSplits = 10` (shallow trees), `NumVariablesToSample = 'all'`

**Examples**:

matlab

```matlab
% Single learner with default settings
Mdl = fitcensemble(X, Y, 'Learners', 'tree')

% Custom tree learner
t = templateTree('MaxNumSplits', 5)
Mdl = fitcensemble(X, Y, 'Learners', t)

% Multiple learner types
t1 = templateTree('MaxNumSplits', 10)
t2 = templateTree('MaxNumSplits', 20)
Mdl = fitcensemble(X, Y, 'Learners', {t1, t2})
```

---

### Performance Optimization

#### **NumBins** - Binning for Speed

- **Purpose**: Speed up training on large datasets
- **Valid when**: Using tree learners
- **Effect**: Bins numeric predictors into equiprobable bins
- **Trade-off**: Faster training, potential accuracy decrease

**Example**:

matlab

```matlab
% Bin numeric predictors into 50 bins
Mdl = fitcensemble(X, Y, 'NumBins', 50)

% Access bin edges after training
edges = Mdl.BinEdges
```

---

#### **Options** - Parallel Computing

matlab

```matlab
Options = statset('UseParallel', true)
Mdl = fitcensemble(X, Y, 'Options', Options)
```

**Requirements**:

- `Method` must be `'Bag'`
- Tree learners only
- Requires Parallel Computing Toolbox

---

### Cross-Validation Options

matlab

```matlab
% 10-fold cross-validation
Mdl = fitcensemble(X, Y, 'CrossVal', 'on')

% 5-fold cross-validation
Mdl = fitcensemble(X, Y, 'KFold', 5)

% Holdout validation (20% holdout)
Mdl = fitcensemble(X, Y, 'Holdout', 0.2)

% Leave-one-out cross-validation
Mdl = fitcensemble(X, Y, 'Leaveout', 'on')

% Custom partition
cvp = cvpartition(500, 'KFold', 5)
Mdl = fitcensemble(X, Y, 'CVPartition', cvp)
```

---

### Classification-Specific Options

#### **ClassNames** - Class Order

matlab

```matlab
% Specify class order
Mdl = fitcensemble(X, Y, 'ClassNames', ["class1", "class2"])
```

#### **Cost** - Misclassification Cost Matrix

matlab

```matlab
% Cost(i,j) = cost of classifying true class i as class j
Cost = [0 1 2; 1 0 2; 2 2 0]
Mdl = fitcensemble(X, Y, 'Cost', Cost)
```

#### **Prior** - Prior Probabilities

matlab

```matlab
% Uniform priors
Mdl = fitcensemble(X, Y, 'Prior', 'uniform')

% Custom priors
Mdl = fitcensemble(X, Y, 'Prior', [0.3 0.7])

% Empirical (default)
Mdl = fitcensemble(X, Y, 'Prior', 'empirical')
```

#### **Weights** - Observation Weights

matlab

```matlab
% Specify weights for each observation
weights = rand(size(Y))
Mdl = fitcensemble(X, Y, 'Weights', weights)
```

---

## Hyperparameter Optimization

### Basic Optimization

matlab

```matlab
% Automatic optimization
Mdl = fitcensemble(X, Y, 'OptimizeHyperparameters', 'auto')

% Optimize all eligible parameters
Mdl = fitcensemble(X, Y, 'OptimizeHyperparameters', 'all')

% Optimize specific parameters
Mdl = fitcensemble(X, Y, 'OptimizeHyperparameters', ...
    {'Method', 'NumLearningCycles', 'LearnRate', 'MinLeafSize'})
```

### Advanced Optimization Options

matlab

```matlab
% Reproducible Bayesian optimization
rng('default')
t = templateTree('Reproducible', true)
Mdl = fitcensemble(X, Y, ...
    'OptimizeHyperparameters', 'auto', ...
    'Learners', t, ...
    'HyperparameterOptimizationOptions', ...
    struct('AcquisitionFunctionName', 'expected-improvement-plus', ...
           'UseParallel', true, ...
           'ShowPlots', true, ...
           'MaxObjectiveEvaluations', 30))
```

### Optimization with Size Constraints

matlab

```matlab
% Optimize model size instead of loss
[Mdl, AggregateResults] = fitcensemble(X, Y, ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', ...
    struct('ConstraintType', 'size', ...
           'ConstraintBounds', [0 1000]))
```

---

## Method-Specific Parameters

### Boosting Methods (AdaBoost, LogitBoost, GentleBoost)

#### **LearnRate** - Shrinkage Parameter

- **Range**: (0, 1]
- **Default**: 1
- **Effect**: Lower values require more learners but often improve accuracy

matlab

```matlab
Mdl = fitcensemble(X, Y, 'Method', 'LogitBoost', 'LearnRate', 0.1)
```

---

### RUSBoost Method

#### **RatioToSmallest** - Sampling Proportion

matlab

```matlab
% Sample 2× the smallest class size from each class
Mdl = fitcensemble(X, Y, 'Method', 'RUSBoost', 'RatioToSmallest', 2)

% Different ratios per class
Mdl = fitcensemble(X, Y, 'Method', 'RUSBoost', 'RatioToSmallest', [2, 1])
```

---

### LPBoost and TotalBoost

#### **MarginPrecision** - Convergence Control

matlab

```matlab
Mdl = fitcensemble(X, Y, 'Method', 'LPBoost', 'MarginPrecision', 0.5)
```

---

### RobustBoost

matlab

```matlab
Mdl = fitcensemble(X, Y, 'Method', 'RobustBoost', ...
    'RobustErrorGoal', 0.05, ...
    'RobustMarginSigma', 0.1, ...
    'RobustMaxMargin', 1)
```

---

### Random Subspace

#### **NPredToSample** - Number of Predictors per Learner

matlab

```matlab
% Sample 5 predictors for each learner
Mdl = fitcensemble(X, Y, 'Method', 'Subspace', 'NPredToSample', 5)

% Train learners for all possible combinations
Mdl = fitcensemble(X, Y, 'Method', 'Subspace', ...
    'NumLearningCycles', 'AllPredictorCombinations', ...
    'NPredToSample', 3)
```

---

### Bagging and Boosting Sampling

matlab

```matlab
% Resample with replacement
Mdl = fitcensemble(X, Y, 'Method', 'Bag', ...
    'Resample', 'on', 'Replace', 'on')

% Sample 75% of data per learner
Mdl = fitcensemble(X, Y, 'Method', 'Bag', ...
    'FResample', 0.75, 'Resample', 'on')
```

---

## Practical Examples

### Example 1: Basic Ensemble Training

matlab

```matlab
% Load data
load census1994

% Train with default settings
Mdl1 = fitcensemble(adultdata, 'salary')

% Train with specific predictors
Mdl2 = fitcensemble(adultdata, 'salary ~ age + education')

% Compare performance
rsLoss1 = resubLoss(Mdl1)  % Full model
rsLoss2 = resubLoss(Mdl2)  % Reduced model
```

---

### Example 2: Speed Optimization with Binning

matlab

```matlab
% Generate large dataset
rng('default')
N = 1e6
X = [mvnrnd([-1 -1], eye(2), N); mvnrnd([1 1], eye(2), N)]
y = [zeros(N, 1); ones(N, 1)]

% Train without binning
tic
Mdl1 = fitcensemble(X, y)
time1 = toc  % ~479 seconds

% Train with binning
tic
Mdl2 = fitcensemble(X, y, 'NumBins', 50)
time2 = toc  % ~166 seconds (3× faster)

% Compare accuracy
loss1 = resubLoss(Mdl1)
loss2 = resubLoss(Mdl2)
```

---

### Example 3: Cross-Validation

matlab

```matlab
load ionosphere

% Create tree template
rng(5)
t = templateTree('MaxNumSplits', 5)

% Train with 10-fold cross-validation
Mdl = fitcensemble(X, Y, ...
    'Method', 'AdaBoostM1', ...
    'Learners', t, ...
    'CrossVal', 'on')

% Plot cumulative loss
kflc = kfoldLoss(Mdl, 'Mode', 'cumulative')
figure
plot(kflc)
ylabel('10-fold Misclassification rate')
xlabel('Learning cycle')

% Get final generalization error
estGenError = kflc(end)
```

---

### Example 4: Hyperparameter Optimization

matlab

```matlab
load ionosphere

% Optimize automatically
rng('default')
t = templateTree('Reproducible', true)
Mdl = fitcensemble(X, Y, ...
    'OptimizeHyperparameters', 'auto', ...
    'Learners', t, ...
    'HyperparameterOptimizationOptions', ...
    struct('AcquisitionFunctionName', 'expected-improvement-plus'))

% View optimization results
Mdl.HyperparameterOptimizationResults
```

---

### Example 5: Manual Cross-Validation Tuning

matlab

```matlab
load ionosphere

rng(1)
n = size(X, 1)
m = floor(log(n - 1) / log(3))
learnRate = [0.1 0.25 0.5 1]
maxNumSplits = 3.^(0:m)
numTrees = 150

% Grid search
numLR = numel(learnRate)
numMNS = numel(maxNumSplits)
Mdl = cell(numMNS, numLR)

for k = 1:numLR
    for j = 1:numMNS
        t = templateTree('MaxNumSplits', maxNumSplits(j))
        Mdl{j,k} = fitcensemble(X, Y, ...
            'NumLearningCycles', numTrees, ...
            'Learners', t, ...
            'KFold', 5, ...
            'LearnRate', learnRate(k))
    end
end

% Compute errors
kflAll = @(x)kfoldLoss(x, 'Mode', 'cumulative')
errorCell = cellfun(kflAll, Mdl, 'Uniform', false)
error = reshape(cell2mat(errorCell), [numTrees numMNS numLR])

% Find optimal parameters
[minErr, minErrIdxLin] = min(error(:))
[idxNumTrees, idxMNS, idxLR] = ind2sub(size(error), minErrIdxLin)

fprintf('Min. error = %0.5f\n', minErr)
fprintf('Optimal: NumTrees=%d, MaxNumSplits=%d, LearnRate=%0.2f\n', ...
    idxNumTrees, maxNumSplits(idxMNS), learnRate(idxLR))

% Train final model
tFinal = templateTree('MaxNumSplits', maxNumSplits(idxMNS))
MdlFinal = fitcensemble(X, Y, ...
    'NumLearningCycles', idxNumTrees, ...
    'Learners', tFinal, ...
    'LearnRate', learnRate(idxLR))
```

---

## Output Objects

### Model Types

|Object Type|When Created|Cross-Validated?|
|---|---|---|
|`ClassificationEnsemble`|No cross-validation, Method ≠ 'Bag'|No|
|`ClassificationBaggedEnsemble`|No cross-validation, Method = 'Bag'|No|
|`ClassificationPartitionedEnsemble`|Cross-validation specified|Yes|

### Accessing Model Properties

matlab

```matlab
% Access trained learners
learners = Mdl.Trained

% Access predictor names
predictorNames = Mdl.PredictorNames

% Access class names
classNames = Mdl.ClassNames

% View bin edges (if NumBins was used)
edges = Mdl.BinEdges
```

---

## Making Predictions

matlab

```matlab
% Predict labels
[labels, scores] = predict(Mdl, Xnew)

% Predict with score transformation
Mdl.ScoreTransform = 'logit'
[labels, scores] = predict(Mdl, Xnew)

% For cross-validated models
[labels, scores] = kfoldPredict(MdlCV)
```

---

## Performance Evaluation

matlab

```matlab
% Resubstitution loss
rsLoss = resubLoss(Mdl)

% Cross-validation loss
cvLoss = kfoldLoss(MdlCV)

% Cumulative loss (for monitoring)
cumulativeLoss = kfoldLoss(MdlCV, 'Mode', 'cumulative')

% Confusion matrix
C = confusionmat(Y, predict(Mdl, X))
```

---

## Best Practices

### 1. **Choose Appropriate Number of Learners**

- Start with 100-200 learners
- Use cross-validation to monitor performance
- Can train incrementally with `resume()` method

### 2. **Tune Both Ensemble and Learner Parameters**

- Default learner parameters may perform poorly
- Use templates to customize learners
- Consider hyperparameter optimization

### 3. **Use Binning for Large Datasets**

- Try `NumBins = 50` as starting point
- Balance speed vs. accuracy
- Only works with tree learners

### 4. **Specify Class Order**

matlab

```matlab
% Determine class order
Ycat = categorical(Y)
classNames = categories(Ycat)

% Use in training
Mdl = fitcensemble(X, Y, 'ClassNames', classNames)
```

### 5. **Handle Imbalanced Data**

- **Method 1**: Adjust misclassification costs

matlab

```matlab
% Higher penalty for minority class
Cost = [0 5; 1 0]  % [TN FP; FN TP]
Mdl = fitcensemble(X, Y, 'Cost', Cost)
```

- **Method 2**: Use RUSBoost

matlab

```matlab
Mdl = fitcensemble(X, Y, 'Method', 'RUSBoost', 'RatioToSmallest', 2)
```

### 6. **Reproducibility**

matlab

```matlab
% Set random seed
rng('default')

% For parallel execution with reproducibility
t = templateTree('Reproducible', true)
Options = statset('UseParallel', true, 'UseSubstreams', true, ...
                 'Streams', RandStream('mlfg6331_64'))
Mdl = fitcensemble(X, Y, 'Learners', t, 'Options', Options)
```

---

## Algorithm Details

### Bagging ('Bag')

1. Draw bootstrap sample (with replacement)
2. Train weak learner on sample
3. Repeat for NumLearningCycles
4. Predict by majority vote

**Key Feature**: Random forest when using trees with random predictor selection

---

### AdaBoost

1. Initialize uniform weights
2. Train weak learner on weighted data
3. Compute weighted error
4. Update weights (increase for misclassified)
5. Repeat, combining learners

**Variants**:

- **AdaBoostM1**: Binary, SAMME algorithm
- **AdaBoostM2**: Multiclass extension

---

### LogitBoost

1. Fit additive logistic regression
2. Use Newton-Raphson steps
3. Minimize logistic loss

---

### RUSBoost

1. Random undersample majority classes
2. Apply AdaBoost to balanced sample
3. Effective for imbalanced data

---

### Random Subspace ('Subspace')

1. Random sample of predictors per learner
2. Train on predictor subset
3. Combine predictions

---

## Common Issues and Solutions

### Issue 1: Slow Training

**Solutions**:

- Use `NumBins` for binning
- Enable parallel processing
- Reduce `NumLearningCycles`
- Use shallower trees

### Issue 2: Overfitting

**Solutions**:

- Reduce tree depth (`MaxNumSplits`)
- Increase `MinLeafSize`
- Use learning rate < 1
- Add more training data

### Issue 3: Poor Cross-Validation Estimates

**Causes**: Imbalanced data, small dataset **Solutions**:

- Balance costs/priors
- Use stratified partitions
- Increase data collection

### Issue 4: Memory Issues

**Solutions**:

- Use compact models: `compact(Mdl)`
- Reduce `NumLearningCycles`
- Use binning
- Process in batches

---

## Code Generation

matlab

```matlab
% Train model
Mdl = fitcensemble(X, Y)

% Generate C/C++ code for prediction
% Requires MATLAB Coder
codegen myPredictFunction -args {Mdl, X}
```

---

## Version Compatibility

- **Introduced**: R2016b
- **Recent Changes**:
    - R2025a: Serial computation fallback for parallel optimization
    - R2023b: Auto optimization includes 'Standardize' for KNN learners

---

## Summary Table: Quick Reference

|Task|Code Example|
|---|---|
|Basic training|`Mdl = fitcensemble(X, Y)`|
|Bagging|`Mdl = fitcensemble(X, Y, 'Method', 'Bag')`|
|Random forest|`Mdl = fitcensemble(X, Y, 'Method', 'Bag')` (with trees)|
|Boosting|`Mdl = fitcensemble(X, Y, 'Method', 'LogitBoost', 'LearnRate', 0.1)`|
|Cross-validation|`Mdl = fitcensemble(X, Y, 'KFold', 5)`|
|Speed optimization|`Mdl = fitcensemble(X, Y, 'NumBins', 50)`|
|Hyperparameter tuning|`Mdl = fitcensemble(X, Y, 'OptimizeHyperparameters', 'auto')`|
|Custom learners|`t = templateTree('MaxNumSplits', 5); Mdl = fitcensemble(X, Y, 'Learners', t)`|
|Parallel training|`Options = statset('UseParallel', true); Mdl = fitcensemble(X, Y, 'Options', Options)`|

---

This tutorial provides comprehensive coverage of `fitcensemble` capabilities for AI coding agents to understand and implement ensemble classification methods in MATLAB.