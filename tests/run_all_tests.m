%% RUN ALL TESTS - WHO Meningioma FTIR Classification Pipeline
% Executes complete test suite with proper path setup
%
% USAGE:
%   run_all_tests

function run_all_tests()
    fprintf('\n');
    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║  WHO MENINGIOMA FTIR CLASSIFICATION - TEST SUITE          ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n');
    fprintf('\n');
    
    % Add paths
    fprintf('Setting up paths...\n');
    addpath(fullfile(pwd, '..', 'src', 'utils'));
    addpath(fullfile(pwd, '..', 'src', 'preprocessing'));
    addpath(fullfile(pwd, '..', 'src', 'classifiers'));
    addpath(fullfile(pwd, '..', 'src', 'validation'));
    addpath(fullfile(pwd, '..', 'src', 'metrics'));
    addpath(fullfile(pwd, '..', 'src', 'reporting'));
    fprintf('✓ Paths configured\n\n');
    
    % Track results
    total_tests = 0;
    passed_tests = 0;
    failed_tests = 0;
    test_results = struct();
    
    % Test suite
    test_files = {
        'test_config.m', ...
        'test_data_loader.m', ...
        'test_preprocessing_pipeline.m', ...
        'test_classifier_wrapper.m', ...
        'test_cross_validation_engine.m', ...
        'test_metrics_calculator.m', ...
        'test_results_aggregator.m'
    };
    
    fprintf('Running %d test files...\n\n', length(test_files));
    fprintf('════════════════════════════════════════════════════════════\n\n');
    
    % Run each test file
    for i = 1:length(test_files)
        test_name = test_files{i};
        fprintf('▶ Running %s...\n', test_name);
        
        try
            % Run test
            run(test_name);
            
            % Record success
            test_results.(strrep(test_name, '.m', '')).status = 'PASS';
            passed_tests = passed_tests + 1;
            fprintf('  ✓ PASSED\n\n');
            
        catch ME
            % Record failure
            test_results.(strrep(test_name, '.m', '')).status = 'FAIL';
            test_results.(strrep(test_name, '.m', '')).error = ME.message;
            failed_tests = failed_tests + 1;
            fprintf('  ✗ FAILED: %s\n\n', ME.message);
        end
        
        total_tests = total_tests + 1;
    end
    
    % Summary
    fprintf('════════════════════════════════════════════════════════════\n');
    fprintf('\n');
    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║  TEST SUITE SUMMARY                                        ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n');
    fprintf('\n');
    fprintf('Total test files:  %d\n', total_tests);
    fprintf('Passed:            %d ✓\n', passed_tests);
    fprintf('Failed:            %d ✗\n', failed_tests);
    fprintf('Success rate:      %.1f%%\n', (passed_tests/total_tests)*100);
    fprintf('\n');
    
    % Individual results
    fprintf('Individual Results:\n');
    fprintf('-------------------\n');
    test_names = fieldnames(test_results);
    for i = 1:length(test_names)
        status = test_results.(test_names{i}).status;
        if strcmp(status, 'PASS')
            fprintf('  ✓ %s: PASS\n', test_names{i});
        else
            fprintf('  ✗ %s: FAIL\n', test_names{i});
            if isfield(test_results.(test_names{i}), 'error')
                fprintf('      Error: %s\n', test_results.(test_names{i}).error);
            end
        end
    end
    fprintf('\n');
    
    % Final verdict
    if failed_tests == 0
        fprintf('╔════════════════════════════════════════════════════════════╗\n');
        fprintf('║  ALL TESTS PASSED ✓✓✓                                     ║\n');
        fprintf('╚════════════════════════════════════════════════════════════╝\n');
    else
        fprintf('╔════════════════════════════════════════════════════════════╗\n');
        fprintf('║  SOME TESTS FAILED - REVIEW ERRORS ABOVE                  ║\n');
        fprintf('╚════════════════════════════════════════════════════════════╝\n');
    end
    fprintf('\n');
end
