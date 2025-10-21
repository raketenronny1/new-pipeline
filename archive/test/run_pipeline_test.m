function run_pipeline_test()
    % Set up paths
    test_dir = fileparts(mfilename('fullpath'));
    pipeline_dir = fileparts(fileparts(test_dir));
    utils_dir = fullfile(pipeline_dir, 'utils');
    
    % Create directories
    if ~exist(utils_dir, 'dir')
        mkdir(utils_dir);
    end
    
    % Add all required directories to path
    addpath(genpath(pipeline_dir));  % Add all subdirectories recursively
    
    % Create test directories
    test_dirs = {'data', 'results', 'models'};
    for i = 1:length(test_dirs)
        test_path = fullfile(test_dir, test_dirs{i});
        if ~exist(test_path, 'dir')
            mkdir(test_path);
        end
    end
    
    fprintf('Directories set up. Pipeline directory: %s\n', pipeline_dir);
    
    % Set up test environment
    setup_test_environment();
    
    % Extract real spectra for testing
    try
        fprintf('Extracting real spectra for testing...\n');
        extract_real_test_data();
    catch ME
        fprintf('Error extracting real test data: %s\n', ME.message);
        fprintf('Make sure the main dataset exists in the data directory.\n');
        return;
    end
    
    % Create test config
    cfg = test_config();
    
    % Run pipeline with test config
    run_full_pipeline(cfg);
    
    % Verify outputs
    verify_outputs();
    
    fprintf('Test run completed.\n');
end

function setup_test_environment()
    fprintf('Setting up test environment...\n');
    
    % Create test directories
    test_dir = fileparts(mfilename('fullpath'));
    test_dirs = {'data', 'results', 'models'};
    for i = 1:length(test_dirs)
        test_path = fullfile(test_dir, test_dirs{i});
        if ~exist(test_path, 'dir')
            mkdir(test_path);
        end
    end
end

function verify_outputs()
    fprintf('\nVerifying pipeline outputs...\n');
    
    % Get test directories
    test_dir = fileparts(mfilename('fullpath'));
    results_dir = fullfile(test_dir, 'results');
    models_dir = fullfile(test_dir, 'models');
    
    % Find the most recent run folder
    result_runs = dir(fullfile(results_dir, 'run_*'));
    if ~isempty(result_runs)
        % Sort by date to get the latest
        [~, idx] = sort([result_runs.datenum], 'descend');
        latest_run = result_runs(idx(1)).name;
        run_results_dir = fullfile(results_dir, latest_run);
        fprintf('Found latest run: %s\n', latest_run);
    else
        run_results_dir = results_dir;
        fprintf('No run-specific folder found. Using base results directory.\n');
    end
    
    % Find the corresponding model folder
    model_runs = dir(fullfile(models_dir, latest_run));
    if ~isempty(model_runs)
        run_models_dir = fullfile(models_dir, latest_run);
    else
        run_models_dir = models_dir;
    end
    
    % Check for expected output files
    expected_files = {
        fullfile(run_results_dir, 'pipeline_log.txt'),
        fullfile(run_results_dir, 'cv_performance.csv'),
        fullfile(run_results_dir, 'test_results.mat'),
        fullfile(run_models_dir, 'final_model.mat')
    };
    
    missing_files = {};
    for i = 1:length(expected_files)
        if ~exist(expected_files{i}, 'file')
            missing_files{end+1} = expected_files{i};
        end
    end
    
    if isempty(missing_files)
        fprintf('✓ All expected output files were generated\n');
    else
        fprintf('⚠ Missing expected files:\n');
        cellfun(@(f) fprintf('  - %s\n', f), missing_files);
    end
    
    % Load and check results
    try
        test_results_path = fullfile(run_results_dir, 'test_results.mat');
        results = load(test_results_path);
        fprintf('✓ Test results loaded successfully\n');
        fprintf('  - Test accuracy: %.2f%%\n', results.test_results.metrics.accuracy * 100);
    catch ME
        fprintf('⚠ Error loading test results: %s\n', ME.message);
    end
end