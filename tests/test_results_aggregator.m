function test_results_aggregator()
    %TEST_RESULTS_AGGREGATOR Unit tests for ResultsAggregator
    %
    % TESTS:
    %   1. Constructor
    %   2. Summarize spectrum-level
    %   3. Summarize patient-level
    %   4. Get best configuration
    %   5. Compare classifiers
    %   6. Export to table
    
    fprintf('\n=== TESTING RESULTS AGGREGATOR ===\n\n');
    
    % Setup
    addpath(fullfile(pwd, '../src/metrics'));
    addpath(fullfile(pwd, '../src/reporting'));
    addpath(fullfile(pwd, '../src/validation'));
    addpath(fullfile(pwd, '../src/classifiers'));
    addpath(fullfile(pwd, '../src/preprocessing'));
    addpath(fullfile(pwd, '../src/utils'));
    
    % Create mock CV results
    cv_results = create_mock_cv_results();
    
    % Test 1: Constructor
    fprintf('Test 1: Constructor... ');
    try
        aggregator = ResultsAggregator(cv_results, 'Verbose', false);
        assert(isa(aggregator, 'ResultsAggregator'), 'Object not created');
        fprintf('PASS\n');
    catch ME
        fprintf('FAIL: %s\n', ME.message);
    end
    
    % Test 2: Summarize spectrum-level
    fprintf('Test 2: Summarize spectrum-level... ');
    try
        aggregator = ResultsAggregator(cv_results, 'Verbose', false);
        summary = aggregator.summarize('Level', 'spectrum');
        
        assert(isfield(summary, 'configurations'), 'Missing configurations');
        assert(size(summary.configurations, 1) == 2, 'Wrong number of permutations');
        assert(size(summary.configurations, 2) == 1, 'Wrong number of classifiers');
        
        % Check first configuration
        config = summary.configurations{1, 1};
        assert(isfield(config, 'mean_metrics'), 'Missing mean_metrics');
        assert(isfield(config, 'std_metrics'), 'Missing std_metrics');
        assert(isfield(config.mean_metrics, 'accuracy'), 'Missing accuracy');
        
        fprintf('PASS\n');
    catch ME
        fprintf('FAIL: %s\n', ME.message);
    end
    
    % Test 3: Summarize patient-level
    fprintf('Test 3: Summarize patient-level... ');
    try
        aggregator = ResultsAggregator(cv_results, 'Verbose', false);
        summary = aggregator.summarize('Level', 'patient');
        
        assert(strcmp(summary.level, 'patient'), 'Wrong level');
        config = summary.configurations{1, 1};
        assert(isfield(config, 'mean_metrics'), 'Missing metrics');
        
        fprintf('PASS\n');
    catch ME
        fprintf('FAIL: %s\n', ME.message);
    end
    
    % Test 4: Get best configuration
    fprintf('Test 4: Get best configuration... ');
    try
        aggregator = ResultsAggregator(cv_results, 'Verbose', false);
        best = aggregator.get_best_configuration('accuracy', 'Level', 'spectrum');
        
        assert(isfield(best, 'permutation_id'), 'Missing permutation_id');
        assert(isfield(best, 'classifier_name'), 'Missing classifier_name');
        assert(isfield(best, 'best_value'), 'Missing best_value');
        assert(best.best_value >= 0 && best.best_value <= 1, 'Invalid accuracy value');
        
        fprintf('PASS\n');
    catch ME
        fprintf('FAIL: %s\n', ME.message);
    end
    
    % Test 5: Compare classifiers
    fprintf('Test 5: Compare classifiers... ');
    try
        aggregator = ResultsAggregator(cv_results, 'Verbose', false);
        comparison = aggregator.compare_classifiers('Metric', 'accuracy', 'Level', 'spectrum');
        
        assert(isfield(comparison, 'classifiers'), 'Missing classifiers');
        assert(isfield(comparison, 'mean_scores'), 'Missing mean_scores');
        assert(strcmp(comparison.metric, 'accuracy'), 'Wrong metric');
        
        fprintf('PASS\n');
    catch ME
        fprintf('FAIL: %s\n', ME.message);
    end
    
    % Test 6: Export to table
    fprintf('Test 6: Export to table... ');
    try
        aggregator = ResultsAggregator(cv_results, 'Verbose', false);
        tbl = aggregator.to_table('Level', 'spectrum');
        
        assert(istable(tbl), 'Not a table');
        assert(height(tbl) == 2, 'Wrong number of rows');
        assert(any(strcmp(tbl.Properties.VariableNames, 'MeanAccuracy')), 'Missing MeanAccuracy column');
        
        fprintf('PASS\n');
    catch ME
        fprintf('FAIL: %s\n', ME.message);
    end
    
    fprintf('\n=== ALL TESTS COMPLETED ===\n');
end

function cv_results = create_mock_cv_results()
    %CREATE_MOCK_CV_RESULTS Create minimal mock CV results for testing
    
    cv_results = struct();
    cv_results.n_samples = 30;
    cv_results.n_features = 10;
    cv_results.n_patients = 12;
    cv_results.n_permutations = 2;
    cv_results.n_classifiers = 1;
    cv_results.n_folds = 3;
    cv_results.n_repeats = 2;
    cv_results.permutations = cell(2, 1);
    
    % Create 2 permutations
    for p = 1:2
        perm = struct();
        perm.permutation_id = sprintf('1020%dX', p-1);
        perm.classifiers = cell(1, 1);
        
        % Create 1 classifier (PCA-LDA)
        clf = struct();
        clf.classifier_name = 'PCA-LDA';
        clf.repeats = cell(2, 1);
        
        % Create 2 repeats
        for r = 1:2
            repeat = struct();
            repeat.folds = cell(3, 1);
            
            % Create 3 folds
            for f = 1:3
                fold = struct();
                fold.repeat = r;
                fold.fold = f;
                fold.n_train_samples = 20;
                fold.n_test_samples = 10;
                fold.n_train_patients = 8;
                fold.n_test_patients = 4;
                
                fold.patient_ids = repelem(1:4, [3 3 2 2])';
                
                % Ensure both classes are present in predictions
                fold.y_true = categorical([ones(5, 1); 2*ones(5, 1)]);
                fold.y_pred = categorical([ones(4, 1); 2*ones(5, 1); 1]);
                fold.scores = [0.9*ones(4, 1), 0.1*ones(4, 1); ...
                              0.1*ones(5, 1), 0.9*ones(5, 1); ...
                              0.6, 0.4];
                
                repeat.folds{f} = fold;
            end
            
            clf.repeats{r} = repeat;
        end
        
        perm.classifiers{1} = clf;
        cv_results.permutations{p} = perm;
    end
end
