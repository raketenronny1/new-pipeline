%% Simplified Test Script for FTIR Pipeline
% This script combines the best working versions of the test pipeline
% Date: October 21, 2025

function run_pipeline_test_simplified()
    % Set up paths
    test_dir = fileparts(mfilename('fullpath'));
    pipeline_dir = fileparts(fileparts(test_dir));
    
    fprintf('Test dir: %s\n', test_dir);
    fprintf('Pipeline dir: %s\n', pipeline_dir);
    
    % Add all required directories to path
    addpath(genpath(pipeline_dir));  % Add all src directory
    
    %% 1. Verify data files exist
    fprintf('Verifying data files...\n');
    data_train_file = fullfile(test_dir, 'data', 'data_table_train.mat');
    data_test_file = fullfile(test_dir, 'data', 'data_table_test.mat');
    wavenumbers_file = fullfile(test_dir, 'data', 'wavenumbers.mat');
    
    % Extract real test data if needed
    if ~exist(data_train_file, 'file') || ~exist(data_test_file, 'file') || ~exist(wavenumbers_file, 'file')
        fprintf('Extracting test data...\n');
        extract_real_test_data();
    end
    
    % Verify data files exist
    if exist(data_train_file, 'file') && exist(data_test_file, 'file') && exist(wavenumbers_file, 'file')
        load(data_train_file, 'dataTableTrain');
        load(data_test_file, 'dataTableTest');
        load(wavenumbers_file, 'wavenumbers_roi');
        
        fprintf('Training set: %d samples\n', size(dataTableTrain, 1));
        fprintf('Test set: %d samples\n', size(dataTableTest, 1));
        fprintf('Wavenumbers: %d points\n', length(wavenumbers_roi));
    else
        error('Required data files not found. Please check the data directory.');
    end
    
    %% 2. Display MATLAB version info
    fprintf('[%s] === MATLAB VERSION INFO ===\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    v = ver;
    for i = 1:length(v)
        fprintf('[%s] %s Version %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), v(i).Name, v(i).Version);
    end
    fprintf('[%s] ========================\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    
    %% 3. Run the full pipeline with test configuration
    try
        % Create test config
        cfg = test_config();
        
        % Run full pipeline
        fprintf('[%s] Starting pipeline execution with test configuration\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
        
        % Phase 0: Quality Control
        fprintf('[%s] Starting Phase 0: Quality Control\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
        quality_control_analysis(cfg);
        
        % Phase 1: Data Loading
        fprintf('[%s] Starting Phase 1: Data Loading\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
        load_and_prepare_data(cfg);
        
        % Phase 2: Feature Selection
        fprintf('[%s] Starting Phase 2: Feature Selection\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
        % Use the no-graphics version of feature selection for batch mode
        perform_feature_selection_fixed_nogfx(cfg);
        
        % Phase 3: Cross-Validation
        fprintf('[%s] Starting Phase 3: Cross-Validation\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
        % Use the fixed cross-validation function that adds optimal_params
        cv_results = run_cross_validation_fixed_with_params(cfg);
        
        % Debug output for cv_results structure
        fprintf('Debug: cv_results structure fields\n');
        for i = 1:length(cv_results)
            fprintf('Classifier %d: %s\n', i, cv_results{i}.classifier);
            fprintf('  Has optimal_params: %d\n', isfield(cv_results{i}, 'optimal_params'));
            if isfield(cv_results{i}, 'optimal_params')
                param_fields = fieldnames(cv_results{i}.optimal_params);
                for j = 1:length(param_fields)
                    fprintf('    %s: %s\n', param_fields{j}, mat2str(cv_results{i}.optimal_params.(param_fields{j})));
                end
            end
        end
        
        % Phase 4: Train Final Model
        fprintf('[%s] Starting Phase 4: Train Final Model\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
        % Use the fixed train_final_model that handles missing optimal_params
        final_model = train_final_model_fixed(cfg, cv_results);
        
        % Phase 5: Test Set Evaluation
        fprintf('[%s] Starting Phase 5: Test Set Evaluation\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
        % Use the no-graphics version for batch mode
        test_results = evaluate_test_set_nogfx(cfg, final_model);
        
        % Phase 6: Generate Report
        fprintf('[%s] Starting Phase 6: Generate Report\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
        % Use the no-graphics version for batch mode
        generate_report_nogfx(cfg, cv_results, final_model, test_results);
        
        fprintf('[%s] Pipeline execution completed successfully!\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
        
    catch ME
        fprintf('\n===== ERROR OCCURRED =====\n');
        fprintf('Error message: %s\n', ME.message);
        
        % Display stack trace
        for i = 1:length(ME.stack)
            fprintf('Function: %s, Line: %d\n', ME.stack(i).name, ME.stack(i).line);
        end
        
        % Check latest results directory
        try
            results_dir = dir(fullfile(cfg.paths.results, 'run_*'));
            if ~isempty(results_dir)
                dates = [results_dir.datenum];
                [~, idx] = max(dates);
                latest_dir = fullfile(cfg.paths.results, results_dir(idx).name);
                
                fprintf('\nLatest results directory: %s\n', latest_dir);
                dir_contents = dir(latest_dir);
                for i = 1:length(dir_contents)
                    fprintf('  %s\n', dir_contents(i).name);
                end
            end
        catch
            fprintf('Could not read latest results directory\n');
        end
        
        % Rethrow error for debugging
        rethrow(ME);
    end
end