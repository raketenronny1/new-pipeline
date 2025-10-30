function test_metrics_calculator()
    %TEST_METRICS_CALCULATOR Unit tests for MetricsCalculator
    %
    % TESTS:
    %   1. Constructor
    %   2. Binary classification metrics
    %   3. Multi-class classification metrics
    %   4. Perfect predictions
    %   5. Confusion matrix correctness
    %   6. Patient-level aggregation (majority vote)
    %   7. AUC calculation
    %   8. Edge cases (single class in predictions)
    
    fprintf('\n=== TESTING METRICS CALCULATOR ===\n\n');
    
    % Setup
    addpath(fullfile(pwd, '../src/metrics'));
    
    % Test 1: Constructor
    fprintf('Test 1: Constructor... ');
    try
        calc = MetricsCalculator('Verbose', false);
        assert(isa(calc, 'MetricsCalculator'), 'Object not created');
        fprintf('PASS\n');
    catch ME
        fprintf('FAIL: %s\n', ME.message);
    end
    
    % Test 2: Binary classification metrics
    fprintf('Test 2: Binary classification metrics... ');
    try
        calc = MetricsCalculator('Verbose', false);
        
        % Create binary test data
        y_true = categorical([1; 1; 1; 1; 2; 2; 2; 2]);
        y_pred = categorical([1; 1; 1; 2; 2; 2; 2; 1]);  % 1 FN, 1 FP
        scores = [0.9 0.1; 0.8 0.2; 0.7 0.3; 0.4 0.6; ...
                  0.2 0.8; 0.1 0.9; 0.3 0.7; 0.6 0.4];
        
        metrics = calc.compute_spectrum_metrics(y_true, y_pred, scores);
        
        % Check basic fields
        assert(isfield(metrics, 'accuracy'), 'Missing accuracy');
        assert(isfield(metrics, 'confusion_matrix'), 'Missing confusion matrix');
        assert(isfield(metrics, 'per_class'), 'Missing per_class');
        
        % Accuracy should be 6/8 = 0.75
        assert(abs(metrics.accuracy - 0.75) < 1e-6, 'Incorrect accuracy');
        
        % Check confusion matrix size
        assert(all(size(metrics.confusion_matrix) == [2 2]), 'Wrong confusion matrix size');
        
        fprintf('PASS\n');
    catch ME
        fprintf('FAIL: %s\n', ME.message);
    end
    
    % Test 3: Multi-class classification metrics
    fprintf('Test 3: Multi-class classification metrics... ');
    try
        calc = MetricsCalculator('Verbose', false);
        
        % Create 3-class test data
        y_true = categorical([1; 1; 1; 2; 2; 2; 3; 3; 3]);
        y_pred = categorical([1; 1; 2; 2; 2; 1; 3; 3; 3]);  % 2 errors
        scores = rand(9, 3);  % Random scores
        
        metrics = calc.compute_spectrum_metrics(y_true, y_pred, scores);
        
        assert(metrics.n_classes == 3, 'Wrong number of classes');
        assert(abs(metrics.accuracy - (7/9)) < 1e-6, 'Incorrect accuracy');
        assert(all(size(metrics.confusion_matrix) == [3 3]), 'Wrong confusion matrix size');
        
        fprintf('PASS\n');
    catch ME
        fprintf('FAIL: %s\n', ME.message);
    end
    
    % Test 4: Perfect predictions
    fprintf('Test 4: Perfect predictions... ');
    try
        calc = MetricsCalculator('Verbose', false);
        
        % Perfect predictions
        y_true = categorical([1; 1; 2; 2; 3; 3]);
        y_pred = categorical([1; 1; 2; 2; 3; 3]);
        scores = [1 0 0; 1 0 0; 0 1 0; 0 1 0; 0 0 1; 0 0 1];
        
        metrics = calc.compute_spectrum_metrics(y_true, y_pred, scores);
        
        assert(metrics.accuracy == 1.0, 'Accuracy should be 1.0');
        assert(metrics.macro_f1 == 1.0, 'F1 should be 1.0');
        assert(metrics.macro_sensitivity == 1.0, 'Sensitivity should be 1.0');
        
        fprintf('PASS\n');
    catch ME
        fprintf('FAIL: %s\n', ME.message);
    end
    
    % Test 5: Confusion matrix correctness
    fprintf('Test 5: Confusion matrix correctness... ');
    try
        calc = MetricsCalculator('Verbose', false);
        
        % Specific case with known confusion matrix
        y_true = categorical([1; 1; 1; 2; 2; 2]);
        y_pred = categorical([1; 1; 2; 1; 2; 2]);  % 1 FN for class 1, 1 FP for class 1
        scores = rand(6, 2);
        
        metrics = calc.compute_spectrum_metrics(y_true, y_pred, scores);
        
        % Expected confusion matrix:
        % [2 1]  <- Row 1: True class 1, predicted 2 once, predicted 1 twice
        % [1 2]  <- Row 2: True class 2, predicted 1 once, predicted 2 twice
        expected_cm = [2 1; 1 2];
        assert(isequal(metrics.confusion_matrix, expected_cm), 'Confusion matrix incorrect');
        
        fprintf('PASS\n');
    catch ME
        fprintf('FAIL: %s\n', ME.message);
    end
    
    % Test 6: Patient-level aggregation
    fprintf('Test 6: Patient-level aggregation (majority vote)... ');
    try
        calc = MetricsCalculator('Verbose', false);
        
        % 2 patients, 3 samples each
        % Patient 1: Class 1 (2 correct, 1 incorrect) -> majority vote = 1
        % Patient 2: Class 2 (1 correct, 2 incorrect) -> majority vote = 1
        patient_ids = [1; 1; 1; 2; 2; 2];
        y_true = categorical([1; 1; 1; 2; 2; 2]);
        y_pred = categorical([1; 1; 2; 1; 1; 2]);
        scores = rand(6, 2);
        
        patient_metrics = calc.compute_patient_metrics(y_true, y_pred, scores, patient_ids);
        
        assert(patient_metrics.n_patients == 2, 'Wrong patient count');
        assert(strcmp(patient_metrics.aggregation_method, 'majority_vote'), ...
            'Wrong aggregation method');
        
        % Patient 1: True=1, Pred=1 (majority) -> Correct
        % Patient 2: True=2, Pred=1 (majority) -> Incorrect
        % Accuracy should be 1/2 = 0.5
        assert(abs(patient_metrics.accuracy - 0.5) < 1e-6, ...
            'Patient-level accuracy incorrect');
        
        fprintf('PASS\n');
    catch ME
        fprintf('FAIL: %s\n', ME.message);
    end
    
    % Test 7: AUC calculation
    fprintf('Test 7: AUC calculation... ');
    try
        calc = MetricsCalculator('Verbose', false);
        
        % Binary case with perfect separation
        y_true = categorical([1; 1; 1; 2; 2; 2]);
        y_pred = categorical([1; 1; 1; 2; 2; 2]);
        scores = [0.9 0.1; 0.8 0.2; 0.7 0.3; 0.1 0.9; 0.2 0.8; 0.3 0.7];
        
        metrics = calc.compute_spectrum_metrics(y_true, y_pred, scores);
        
        % Perfect classification should have AUC = 1.0
        assert(isfield(metrics, 'auc'), 'Missing AUC field');
        assert(metrics.auc == 1.0, sprintf('AUC should be 1.0, got %.4f', metrics.auc));
        
        fprintf('PASS\n');
    catch ME
        fprintf('FAIL: %s\n', ME.message);
    end
    
    % Test 8: Edge case - all same prediction
    fprintf('Test 8: Edge case (all same prediction)... ');
    try
        calc = MetricsCalculator('Verbose', false);
        
        % All predicted as class 1
        y_true = categorical([1; 1; 2; 2; 3; 3]);
        y_pred = categorical([1; 1; 1; 1; 1; 1]);
        scores = repmat([1 0 0], 6, 1);
        
        metrics = calc.compute_spectrum_metrics(y_true, y_pred, scores);
        
        % Should handle gracefully (some metrics will be NaN)
        assert(~isempty(metrics), 'Metrics calculation failed');
        assert(isfield(metrics, 'accuracy'), 'Missing accuracy');
        
        fprintf('PASS\n');
    catch ME
        fprintf('FAIL: %s\n', ME.message);
    end
    
    fprintf('\n=== ALL TESTS COMPLETED ===\n');
end
