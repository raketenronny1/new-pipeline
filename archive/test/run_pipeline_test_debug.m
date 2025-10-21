function run_pipeline_test_debug()
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
    
    % Extract real spectra for testing if not already done
    if ~exist(fullfile(test_dir, 'data', 'data_table_train.mat'), 'file')
        try
            fprintf('Extracting real spectra for testing...\n');
            extract_real_test_data();
        catch ME
            fprintf('Error extracting real test data: %s\n', ME.message);
            fprintf('Make sure the main dataset exists in the data directory.\n');
            return;
        end
    else
        fprintf('Using existing test data files...\n');
    end
    
    % Create test config
    try
        cfg = test_config();
        
        % Print key configuration values for debugging
        fprintf('\n=== TEST CONFIGURATION ===\n');
        fprintf('Data path: %s\n', cfg.paths.data);
        fprintf('Results path: %s\n', cfg.paths.results);
        fprintf('Models path: %s\n', cfg.paths.models);
        fprintf('QC path: %s\n', cfg.paths.qc);
        fprintf('SNR threshold: %.2f\n', cfg.qc.snr_threshold);
        fprintf('CV folds: %d, repeats: %d\n', cfg.cv.n_folds, cfg.cv.n_repeats);
    catch ME
        fprintf('Error creating config: %s\n', ME.message);
        return;
    end
    
    % Load sample data to verify it exists
    try
        fprintf('\nVerifying data files...\n');
        load(fullfile(cfg.paths.data, 'data_table_train.mat'), 'dataTableTrain');
        load(fullfile(cfg.paths.data, 'data_table_test.mat'), 'dataTableTest');
        load(fullfile(cfg.paths.data, 'wavenumbers.mat'), 'wavenumbers_roi');
        
        fprintf('Training set: %d samples\n', height(dataTableTrain));
        fprintf('Test set: %d samples\n', height(dataTableTest));
        fprintf('Wavenumbers: %d points\n', length(wavenumbers_roi));
        
        % Check the first sample's spectra
        if ~isempty(dataTableTrain.CombinedSpectra{1})
            [n_spectra, n_wavenumbers] = size(dataTableTrain.CombinedSpectra{1});
            fprintf('First train sample has %d spectra with %d wavenumbers each\n', n_spectra, n_wavenumbers);
            
            % Check for NaN or Inf values in the first sample
            if any(any(isnan(dataTableTrain.CombinedSpectra{1}))) || any(any(isinf(dataTableTrain.CombinedSpectra{1})))
                fprintf('WARNING: First train sample contains NaN or Inf values!\n');
            else
                fprintf('First train sample contains valid data.\n');
            end
        else
            fprintf('WARNING: First train sample has no spectra!\n');
        end
    catch ME
        fprintf('Error loading data files: %s\n', ME.message);
        return;
    end
    
    % Use the fixed versions of functions for testing
    fprintf('Using fixed versions of functions for testing...\n');
    
    % Create backup of original functions if they don't exist
    if ~exist([pipeline_dir filesep 'perform_feature_selection_orig.m'], 'file')
        copyfile([pipeline_dir filesep 'perform_feature_selection.m'], ...
                [pipeline_dir filesep 'perform_feature_selection_orig.m']);
    end
    
    if ~exist([pipeline_dir filesep 'load_and_prepare_data_orig.m'], 'file')
        copyfile([pipeline_dir filesep 'load_and_prepare_data.m'], ...
                [pipeline_dir filesep 'load_and_prepare_data_orig.m']);
    end
    
    % Copy the fixed versions to use
    copyfile([test_dir filesep 'perform_feature_selection_fixed.m'], ...
             [pipeline_dir filesep 'perform_feature_selection.m']);
    
    copyfile([test_dir filesep 'load_and_prepare_data_fixed.m'], ...
             [pipeline_dir filesep 'load_and_prepare_data.m']);
             
    % Run pipeline with test config in a try-catch block
    try
        run_full_pipeline(cfg);
    catch ME
        fprintf('Error running pipeline: %s\n', ME.message);
        fprintf('Error details: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
        
        if length(ME.stack) > 1
            fprintf('\nCall stack:\n');
            for i = 1:min(5, length(ME.stack))
                fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
            end
        end
        
        % Restore original files
        copyfile([pipeline_dir filesep 'perform_feature_selection_orig.m'], ...
                 [pipeline_dir filesep 'perform_feature_selection.m']);
        
        copyfile([pipeline_dir filesep 'load_and_prepare_data_orig.m'], ...
                 [pipeline_dir filesep 'load_and_prepare_data.m']);
                 
        return;
    end
    
    % Verify outputs
    verify_outputs();
    
    % Restore original files if they exist
    if exist([pipeline_dir filesep 'perform_feature_selection_orig.m'], 'file')
        copyfile([pipeline_dir filesep 'perform_feature_selection_orig.m'], ...
                 [pipeline_dir filesep 'perform_feature_selection.m']);
    end
    
    if exist([pipeline_dir filesep 'load_and_prepare_data_orig.m'], 'file')
        copyfile([pipeline_dir filesep 'load_and_prepare_data_orig.m'], ...
                 [pipeline_dir filesep 'load_and_prepare_data.m']);
    end
    
    fprintf('Test run completed.\n');
end