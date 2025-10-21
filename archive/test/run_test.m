%% Script to manually run and debug the test pipeline
% This script uses the consolidated working versions of the pipeline functions
% Last updated: October 21, 2025

% Set up paths
test_dir = pwd;
pipeline_dir = fileparts(test_dir); % Path to meningioma_ftir_pipeline
disp(['Test dir: ' test_dir]);
disp(['Pipeline dir: ' pipeline_dir]);

% Add all required directories to path
addpath(genpath(fileparts(pipeline_dir))); % Add all src directory

% Copy the fixed versions to use if needed
if exist([test_dir filesep 'perform_feature_selection_fixed.m'], 'file') 
    copyfile([test_dir filesep 'perform_feature_selection_fixed.m'], ...
             [pipeline_dir filesep 'perform_feature_selection.m']);
    disp('Copied fixed perform_feature_selection.m');
else
    disp('Warning: perform_feature_selection_fixed.m not found, using current version');
end

if exist([test_dir filesep 'load_and_prepare_data_fixed.m'], 'file')
    copyfile([test_dir filesep 'load_and_prepare_data_fixed.m'], ...
             [pipeline_dir filesep 'load_and_prepare_data.m']);
    disp('Copied fixed load_and_prepare_data.m');
else
    disp('Warning: load_and_prepare_data_fixed.m not found, using current version');
end

% Backup and replace run_cross_validation.m
if exist([pipeline_dir filesep 'run_cross_validation.m'], 'file')
    if ~exist([pipeline_dir filesep 'run_cross_validation_orig.m'], 'file')
        copyfile([pipeline_dir filesep 'run_cross_validation.m'], ...
                [pipeline_dir filesep 'run_cross_validation_orig.m']);
        disp('Backed up run_cross_validation.m');
    end
    
    % Create a copy of the fixed run_cross_validation function
    copyfile([pipeline_dir filesep 'run_cross_validation_orig.m'], ...
             [test_dir filesep 'run_cross_validation_complete.m']);
    disp('Created complete run_cross_validation_complete.m');
    
    % Create a fixed version of run_cross_validation.m manually
    fixedContent = [
        '%% PHASE 3: MODEL SELECTION VIA CROSS-VALIDATION' newline ...
        '% This script performs model selection using cross-validation on the training set' newline ...
        '' newline ...
        'function run_cross_validation(cfg)' newline ...
        '    % Input validation' newline ...
        '    if ~isstruct(cfg) || ~isfield(cfg, ''paths'') || ~isfield(cfg.paths, ''results'')' newline ...
        '        error(''Invalid cfg structure. Must contain paths.results'');' newline ...
        '    end' newline ...
        '' newline ...
        '    %% Load Data' newline ...
        '    fprintf(''Loading transformed training data...\\n'');' newline ...
        '    load(fullfile(cfg.paths.results, ''X_train_pca.mat''), ''X_train_pca'');' newline ...
        '    load(fullfile(cfg.paths.results, ''preprocessed_data.mat''), ''trainingData'');' newline ...
        '' newline ...
        '    %% Set Up Cross-Validation' newline ...
        '    fprintf(''Setting up cross-validation...\\n'');' newline ...
        '' newline ...
        '    % Set random seed from config or use default' newline ...
        '    if isfield(cfg, ''random_seed'')' newline ...
        '        rng(cfg.random_seed, ''twister'');' newline ...
        '    else' newline ...
        '        rng(42, ''twister'');' newline ...
        '        warning(''No random seed specified in cfg. Using default seed 42.'');' newline ...
        '    end' newline ...
        '' newline ...
        '    % CV parameters - use config values or defaults' newline ...
        '    if isfield(cfg, ''cv'') && isfield(cfg.cv, ''n_folds'')' newline ...
        '        n_folds = cfg.cv.n_folds;' newline ...
        '        fprintf(''Using %d folds from configuration\\n'', n_folds);' newline ...
        '    else' newline ...
        '        n_folds = 5;' newline ...
        '        fprintf(''Using default %d folds\\n'', n_folds);' newline ...
        '    end' newline ...
        '    ' newline ...
        '    if isfield(cfg, ''cv'') && isfield(cfg.cv, ''n_repeats'')' newline ...
        '        n_repeats = cfg.cv.n_repeats;' newline ...
        '        fprintf(''Using %d repeats from configuration\\n'', n_repeats);' newline ...
        '    else' newline ...
        '        n_repeats = 50;' newline ...
        '        fprintf(''Using default %d repeats\\n'', n_repeats);' newline ...
        '    end' newline ...
        '' newline ...
        '    % Print CV settings' newline ...
        '    fprintf(''Cross-validation settings: %d folds, %d repeats\\n'', n_folds, n_repeats);' newline ...
    ];
    
    % Get the rest of the original file after the hardcoded n_folds and n_repeats
    originalFile = fileread([pipeline_dir filesep 'run_cross_validation_orig.m']);
    
    % Find the position right after n_repeats = 50
    startPos = strfind(originalFile, 'n_repeats = 50;');
    if ~isempty(startPos)
        startPos = startPos + length('n_repeats = 50;');
        % Find the next newline
        newlinePos = min(strfind(originalFile(startPos:end), newline));
        if ~isempty(newlinePos)
            startPos = startPos + newlinePos;
            remainingText = originalFile(startPos:end);
            
            % Create the complete fixed file
            fileID = fopen([test_dir filesep 'run_cross_validation_fixed.m'], 'w');
            if fileID ~= -1
                fprintf(fileID, '%s', fixedContent);
                fprintf(fileID, '%s', remainingText);
                fclose(fileID);
                disp('Created fixed run_cross_validation_fixed.m');
                
                % Copy the fixed file to the pipeline directory
                copyfile([test_dir filesep 'run_cross_validation_fixed.m'], ...
                        [pipeline_dir filesep 'run_cross_validation.m']);
                disp('Copied fixed run_cross_validation.m to pipeline directory');
            else
                disp('ERROR: Could not create run_cross_validation_fixed.m');
            end
        else
            disp('ERROR: Could not find newline after n_repeats');
        end
    else
        disp('ERROR: Could not find n_repeats in original file');
    end
else
    disp('WARNING: run_cross_validation.m not found in pipeline directory');
end

% Create test config
cfg = test_config();

% Verify data files
disp('Verifying data files...');
try
    load(fullfile(cfg.paths.data, 'data_table_train.mat'), 'dataTableTrain');
    load(fullfile(cfg.paths.data, 'data_table_test.mat'), 'dataTableTest');
    load(fullfile(cfg.paths.data, 'wavenumbers.mat'), 'wavenumbers_roi');
    disp(['Training set: ' num2str(height(dataTableTrain)) ' samples']);
    disp(['Test set: ' num2str(height(dataTableTest)) ' samples']);
    disp(['Wavenumbers: ' num2str(length(wavenumbers_roi)) ' points']);
catch ME
    disp(['Error loading data: ' ME.message]);
    return;
end

% Run the full pipeline
try
    run_full_pipeline(cfg);
    disp('Pipeline completed successfully!');
catch ME
    disp(['Error running pipeline: ' ME.message]);
    disp(['Error details: ' ME.stack(1).name ' (line ' num2str(ME.stack(1).line) ')']);
    
    if length(ME.stack) > 1
        disp('Call stack:');
        for i = 1:min(5, length(ME.stack))
            disp(['  ' ME.stack(i).name ' (line ' num2str(ME.stack(i).line) ')']);
        end
    end
end

% Restore original files
try
    copyfile([pipeline_dir filesep 'perform_feature_selection_orig.m'], ...
             [pipeline_dir filesep 'perform_feature_selection.m']);
    disp('Restored original perform_feature_selection.m');

    copyfile([pipeline_dir filesep 'load_and_prepare_data_orig.m'], ...
             [pipeline_dir filesep 'load_and_prepare_data.m']);
    disp('Restored original load_and_prepare_data.m');
catch ME
    disp(['Error restoring files: ' ME.message]);
end

disp('Test script completed.');