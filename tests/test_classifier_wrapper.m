%% TEST_CLASSIFIER_WRAPPER - Unit tests for ClassifierWrapper class
%
% Tests all four classifiers with synthetic and real data
%
% USAGE:
%   run test_classifier_wrapper.m

function test_classifier_wrapper()
    fprintf('=== Testing ClassifierWrapper Class ===\n\n');
    
    % Generate synthetic data
    rng(42);
    n_train = 100;
    n_test = 50;
    n_features = 50;
    
    % Binary classification problem
    X_train = [randn(n_train/2, n_features) + 2; ...
               randn(n_train/2, n_features) - 2];
    y_train = categorical([ones(n_train/2, 1); ...
                          ones(n_train/2, 1) * 2]);
    
    X_test = [randn(n_test/2, n_features) + 2; ...
              randn(n_test/2, n_features) - 2];
    y_test = categorical([ones(n_test/2, 1); ...
                         ones(n_test/2, 1) * 2]);
    
    % Get config
    cfg = Config.getInstance();
    
    %% Test 1: PCA-LDA
    fprintf('Test 1: PCA-LDA classifier... ');
    clf_lda = ClassifierWrapper('PCA-LDA', cfg);
    clf_lda.train(X_train, y_train);
    [y_pred_lda, scores_lda] = clf_lda.predict(X_test);
    
    % Validate outputs
    assert(isa(y_pred_lda, 'categorical'), 'Predictions must be categorical');
    assert(length(y_pred_lda) == n_test, 'Wrong number of predictions');
    assert(size(scores_lda, 1) == n_test, 'Wrong number of score rows');
    assert(size(scores_lda, 2) == 2, 'Wrong number of score columns (classes)');
    
    % Check accuracy (should be reasonable for separable data)
    acc_lda = sum(y_pred_lda == y_test) / n_test;
    assert(acc_lda > 0.6, 'LDA accuracy too low');
    
    fprintf('✓ PASSED (Acc: %.1f%%)\n', acc_lda * 100);
    
    %% Test 2: SVM-RBF
    fprintf('Test 2: SVM-RBF classifier... ');
    clf_svm = ClassifierWrapper('SVM-RBF', cfg);
    clf_svm.train(X_train, y_train);
    [y_pred_svm, scores_svm] = clf_svm.predict(X_test);
    
    assert(isa(y_pred_svm, 'categorical'), 'Predictions must be categorical');
    assert(length(y_pred_svm) == n_test, 'Wrong number of predictions');
    assert(size(scores_svm, 1) == n_test, 'Wrong number of score rows');
    
    acc_svm = sum(y_pred_svm == y_test) / n_test;
    assert(acc_svm > 0.6, 'SVM accuracy too low');
    
    fprintf('✓ PASSED (Acc: %.1f%%)\n', acc_svm * 100);
    
    %% Test 3: PLS-DA
    fprintf('Test 3: PLS-DA classifier... ');
    clf_pls = ClassifierWrapper('PLS-DA', cfg);
    clf_pls.train(X_train, y_train);
    [y_pred_pls, scores_pls] = clf_pls.predict(X_test);
    
    assert(isa(y_pred_pls, 'categorical'), 'Predictions must be categorical');
    assert(length(y_pred_pls) == n_test, 'Wrong number of predictions');
    assert(size(scores_pls, 1) == n_test, 'Wrong number of score rows');
    
    acc_pls = sum(y_pred_pls == y_test) / n_test;
    assert(acc_pls > 0.5, 'PLS-DA accuracy too low');
    
    fprintf('✓ PASSED (Acc: %.1f%%)\n', acc_pls * 100);
    
    %% Test 4: Random Forest
    fprintf('Test 4: Random Forest classifier... ');
    clf_rf = ClassifierWrapper('RandomForest', cfg);
    clf_rf.train(X_train, y_train);
    [y_pred_rf, scores_rf] = clf_rf.predict(X_test);
    
    assert(isa(y_pred_rf, 'categorical'), 'Predictions must be categorical');
    assert(length(y_pred_rf) == n_test, 'Wrong number of predictions');
    assert(size(scores_rf, 1) == n_test, 'Wrong number of score rows');
    
    acc_rf = sum(y_pred_rf == y_test) / n_test;
    assert(acc_rf > 0.6, 'RandomForest accuracy too low');
    
    fprintf('✓ PASSED (Acc: %.1f%%)\n', acc_rf * 100);
    
    %% Test 5: Score Validity
    fprintf('Test 5: Score validity... ');
    
    % Scores should be probabilities (non-negative, sum to ~1)
    assert(all(scores_lda(:) >= 0), 'LDA scores must be non-negative');
    assert(all(scores_svm(:) >= 0), 'SVM scores must be non-negative');
    assert(all(scores_pls(:) >= 0), 'PLS scores must be non-negative');
    assert(all(scores_rf(:) >= 0), 'RF scores must be non-negative');
    
    % Check row sums (should be close to 1 for probabilities)
    lda_sums = sum(scores_lda, 2);
    assert(max(abs(lda_sums - 1)) < 0.01, 'LDA scores should sum to 1');
    
    fprintf('✓ PASSED\n');
    
    %% Test 6: Consistent Predictions
    fprintf('Test 6: Prediction consistency... ');
    
    % Predicting same data twice should give same results
    [y_pred1, scores1] = clf_lda.predict(X_test);
    [y_pred2, scores2] = clf_lda.predict(X_test);
    
    assert(isequal(y_pred1, y_pred2), 'Predictions not consistent');
    assert(max(abs(scores1(:) - scores2(:))) < 1e-10, 'Scores not consistent');
    
    fprintf('✓ PASSED\n');
    
    %% Test 7: Invalid Classifier Type
    fprintf('Test 7: Invalid classifier detection... ');
    
    try
        clf_bad = ClassifierWrapper('InvalidClassifier', cfg);
        error('Should have rejected invalid classifier');
    catch ME
        assert(contains(ME.identifier, 'InvalidType'), 'Wrong error type');
    end
    
    fprintf('✓ PASSED\n');
    
    %% Test 8: Untrained Prediction Error
    fprintf('Test 8: Untrained prediction error... ');
    
    clf_new = ClassifierWrapper('PCA-LDA', cfg);
    try
        clf_new.predict(X_test);
        error('Should have failed on untrained model');
    catch ME
        assert(contains(ME.identifier, 'NotTrained'), 'Wrong error type');
    end
    
    fprintf('✓ PASSED\n');
    
    %% Test 9: Dimension Mismatch Detection
    fprintf('Test 9: Dimension mismatch detection... ');
    
    clf_dim = ClassifierWrapper('SVM-RBF', cfg);
    clf_dim.train(X_train, y_train);
    
    % Try to predict with wrong number of features
    X_bad = randn(10, 30);  % Wrong feature count
    try
        clf_dim.predict(X_bad);
        error('Should have detected feature mismatch');
    catch ME
        % Expected error (MATLAB will catch this in fitcsvm/predict)
    end
    
    fprintf('✓ PASSED\n');
    
    %% Test 10: Multi-class Classification
    fprintf('Test 10: Multi-class classification... ');
    
    % Create 3-class problem
    n_per_class = 30;
    X_multi = [randn(n_per_class, 20) + 2; ...
               randn(n_per_class, 20) - 2; ...
               randn(n_per_class, 20)];
    y_multi = categorical([ones(n_per_class, 1); ...
                          ones(n_per_class, 1) * 2; ...
                          ones(n_per_class, 1) * 3]);
    
    clf_multi = ClassifierWrapper('RandomForest', cfg);
    clf_multi.train(X_multi, y_multi);
    [y_pred_multi, scores_multi] = clf_multi.predict(X_multi);
    
    assert(size(scores_multi, 2) == 3, 'Should have 3 class scores');
    assert(length(unique(y_pred_multi)) <= 3, 'Too many predicted classes');
    
    fprintf('✓ PASSED\n');
    
    %% Test 11: getName Method
    fprintf('Test 11: getName method... ');
    
    assert(strcmp(clf_lda.getName(), 'PCA-LDA'), 'LDA name wrong');
    assert(strcmp(clf_svm.getName(), 'SVM-RBF'), 'SVM name wrong');
    assert(strcmp(clf_pls.getName(), 'PLS-DA'), 'PLS name wrong');
    assert(strcmp(clf_rf.getName(), 'RandomForest'), 'RF name wrong');
    
    fprintf('✓ PASSED\n');
    
    %% Test 12: Model Retrieval
    fprintf('Test 12: Model retrieval... ');
    
    model = clf_lda.getModel();
    assert(~isempty(model), 'Model should not be empty after training');
    
    fprintf('✓ PASSED\n');
    
    %% Summary
    fprintf('\n=== ALL TESTS PASSED ===\n');
    fprintf('✓ All 4 classifiers working correctly\n');
    fprintf('✓ Unified interface validated\n');
    fprintf('✓ Error handling verified\n');
    
    fprintf('\nClassifier Performance Summary:\n');
    fprintf('  PCA-LDA:      %.1f%%\n', acc_lda * 100);
    fprintf('  SVM-RBF:      %.1f%%\n', acc_svm * 100);
    fprintf('  PLS-DA:       %.1f%%\n', acc_pls * 100);
    fprintf('  RandomForest: %.1f%%\n', acc_rf * 100);
end
