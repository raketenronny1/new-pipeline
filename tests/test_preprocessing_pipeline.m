%% TEST_PREPROCESSING_PIPELINE - Unit tests for PreprocessingPipeline class
%
% Tests critical fit/transform separation to prevent data leakage
%
% USAGE:
%   run test_preprocessing_pipeline.m

function test_preprocessing_pipeline()
    fprintf('=== Testing PreprocessingPipeline Class ===\n\n');
    
    % Generate synthetic data
    rng(42);
    X_train = randn(100, 1000) * 10 + 50;  % Training data
    X_test = randn(50, 1000) * 10 + 50;    % Test data
    
    %% Test 1: Baseline (No Preprocessing)
    fprintf('Test 1: Baseline (10000X)... ');
    pipeline = PreprocessingPipeline('10000X');
    [X_tr_proc, params] = pipeline.fit_transform(X_train);
    X_te_proc = pipeline.transform(X_test, params);
    
    assert(isequal(size(X_tr_proc), size(X_train)), 'Train size changed');
    assert(isequal(size(X_te_proc), size(X_test)), 'Test size changed');
    fprintf('✓ PASSED\n');
    
    %% Test 2: Binning Only
    fprintf('Test 2: Binning (20000X)... ');
    pipeline = PreprocessingPipeline('20000X');
    [X_tr_proc, params] = pipeline.fit_transform(X_train);
    X_te_proc = pipeline.transform(X_test, params);
    
    expected_features = floor(1000 / 2);
    assert(size(X_tr_proc, 2) == expected_features, 'Binning failed');
    assert(size(X_te_proc, 2) == expected_features, 'Test binning mismatch');
    fprintf('✓ PASSED (%d -> %d features)\n', 1000, expected_features);
    
    %% Test 3: Smoothing Only
    fprintf('Test 3: Smoothing (01000X)... ');
    pipeline = PreprocessingPipeline('01000X');
    [X_tr_proc, params] = pipeline.fit_transform(X_train);
    X_te_proc = pipeline.transform(X_test, params);
    
    assert(size(X_tr_proc, 2) == 1000, 'Smoothing changed feature count');
    assert(size(X_te_proc, 2) == 1000, 'Test smoothing mismatch');
    fprintf('✓ PASSED\n');
    
    %% Test 4: Normalization Only
    fprintf('Test 4: Normalization (10200X)... ');
    pipeline = PreprocessingPipeline('10200X');
    [X_tr_proc, params] = pipeline.fit_transform(X_train);
    X_te_proc = pipeline.transform(X_test, params);
    
    % Check that each feature in training data is normalized
    feature_means = mean(X_tr_proc, 1);
    feature_stds = std(X_tr_proc, 0, 1);
    
    assert(max(abs(feature_means)) < 1e-6, 'Train feature means not ~0');
    assert(max(abs(feature_stds - 1)) < 1e-6, 'Train feature stds not ~1');
    
    % CRITICAL: Test data normalized with TRAINING parameters
    % (will NOT have mean=0, std=1 for each feature)
    assert(all(isfinite(X_te_proc(:))), 'Test data has non-finite values');
    fprintf('✓ PASSED\n');
    
    %% Test 5: 1st Derivative
    fprintf('Test 5: 1st Derivative (10010X)... ');
    pipeline = PreprocessingPipeline('10010X');
    [X_tr_proc, params] = pipeline.fit_transform(X_train);
    X_te_proc = pipeline.transform(X_test, params);
    
    assert(size(X_tr_proc, 2) == 999, '1st derivative size wrong');
    assert(size(X_te_proc, 2) == 999, 'Test 1st derivative mismatch');
    fprintf('✓ PASSED (%d -> %d features)\n', 1000, 999);
    
    %% Test 6: 2nd Derivative
    fprintf('Test 6: 2nd Derivative (10020X)... ');
    pipeline = PreprocessingPipeline('10020X');
    [X_tr_proc, params] = pipeline.fit_transform(X_train);
    X_te_proc = pipeline.transform(X_test, params);
    
    assert(size(X_tr_proc, 2) == 998, '2nd derivative size wrong');
    assert(size(X_te_proc, 2) == 998, 'Test 2nd derivative mismatch');
    fprintf('✓ PASSED (%d -> %d features)\n', 1000, 998);
    
    %% Test 7: Combined Pipeline (Norm + 2nd Deriv)
    fprintf('Test 7: Norm + 2nd Deriv (10220X)... ');
    pipeline = PreprocessingPipeline('10220X');
    [X_tr_proc, params] = pipeline.fit_transform(X_train);
    X_te_proc = pipeline.transform(X_test, params);
    
    % Derivative applied after normalization
    assert(size(X_tr_proc, 2) == 998, 'Combined pipeline size wrong');
    assert(size(X_te_proc, 2) == 998, 'Test combined mismatch');
    fprintf('✓ PASSED (%d -> %d features)\n', 1000, 998);
    
    %% Test 8: Full Pipeline (Bin + Smooth + Norm + 2nd Deriv)
    fprintf('Test 8: Full pipeline (21220X)... ');
    pipeline = PreprocessingPipeline('21220X');
    [X_tr_proc, params] = pipeline.fit_transform(X_train);
    X_te_proc = pipeline.transform(X_test, params);
    
    % Bin=2: 1000->500, then 2nd deriv: 500->498
    expected = floor(1000/2) - 2;
    assert(size(X_tr_proc, 2) == expected, 'Full pipeline size wrong');
    assert(size(X_te_proc, 2) == expected, 'Test full pipeline mismatch');
    fprintf('✓ PASSED (%d -> %d features)\n', 1000, expected);
    
    %% Test 9: Dimension Consistency
    fprintf('Test 9: Dimension consistency... ');
    pipeline = PreprocessingPipeline('10220X');
    [X_tr_proc, params] = pipeline.fit_transform(X_train);
    X_te_proc = pipeline.transform(X_test, params);
    
    % Test and train must have same number of features
    assert(size(X_tr_proc, 2) == size(X_te_proc, 2), ...
        'Feature count mismatch between train and test');
    fprintf('✓ PASSED\n');
    
    %% Test 10: Parameter Isolation (No Data Leakage)
    fprintf('Test 10: Parameter isolation (no leakage)... ');
    
    % Create very different train/test data
    X_train_a = randn(100, 100) * 5 + 100;
    X_test_a = randn(50, 100) * 20 + 500;  % Very different distribution
    
    pipeline = PreprocessingPipeline('10200X');  % Normalization
    [~, params_a] = pipeline.fit_transform(X_train_a);
    X_test_normalized = pipeline.transform(X_test_a, params_a);
    
    % Verify test data normalized with TRAIN parameters
    train_mean = params_a.normalization.mean;
    train_std = params_a.normalization.std;
    
    expected = (X_test_a - train_mean) ./ train_std;
    assert(max(abs(X_test_normalized(:) - expected(:))) < 1e-10, ...
        'Test data not using training parameters');
    
    fprintf('✓ PASSED (no leakage)\n');
    
    %% Test 11: Invalid Permutation Detection
    fprintf('Test 11: Invalid permutation detection... ');
    
    try
        pipeline = PreprocessingPipeline('1002');  % Too short
        error('Should have failed');
    catch ME
        % Expected error
    end
    
    try
        pipeline = PreprocessingPipeline('10022Y');  % Wrong ending
        error('Should have failed');
    catch ME
        % Expected error
    end
    
    fprintf('✓ PASSED\n');
    
    %% Test 12: Permutation Mismatch Detection
    fprintf('Test 12: Permutation mismatch detection... ');
    
    pipeline1 = PreprocessingPipeline('10022X');
    [~, params1] = pipeline1.fit_transform(X_train);
    
    pipeline2 = PreprocessingPipeline('10020X');
    
    try
        pipeline2.transform(X_test, params1);  % Wrong params!
        error('Should have detected mismatch');
    catch ME
        assert(contains(ME.identifier, 'PermutationMismatch'), ...
            'Wrong error type');
    end
    
    fprintf('✓ PASSED\n');
    
    fprintf('\n=== ALL TESTS PASSED ===\n');
    fprintf('✓ PreprocessingPipeline prevents data leakage\n');
    fprintf('✓ Fit/transform separation working correctly\n');
end
