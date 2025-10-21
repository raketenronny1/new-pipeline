%% Cleanup Script for FTIR Pipeline Test Directory
% This script will consolidate the test workflow and clean up redundant files
% Date: October 21, 2025

% Set paths
test_dir = pwd;
pipeline_dir = fileparts(test_dir);
project_root = fileparts(fileparts(test_dir));

fprintf('Starting cleanup process...\n');
fprintf('Test directory: %s\n', test_dir);
fprintf('Pipeline directory: %s\n', pipeline_dir);
fprintf('Project root: %s\n', project_root);

%% 1. Create backup of current pipeline files
backup_dir = fullfile(project_root, 'backup_pre_cleanup_20251021');
if ~exist(backup_dir, 'dir')
    mkdir(backup_dir);
    fprintf('Created backup directory: %s\n', backup_dir);
    
    % Copy pipeline files to backup
    pipeline_files = dir(fullfile(pipeline_dir, '*.m'));
    for i = 1:length(pipeline_files)
        copyfile(fullfile(pipeline_dir, pipeline_files(i).name), ...
                 fullfile(backup_dir, pipeline_files(i).name));
    end
    fprintf('Copied %d pipeline files to backup\n', length(pipeline_files));
else
    fprintf('Backup directory already exists. Skipping backup.\n');
end

%% 2. Consolidate working versions - replace main pipeline files with fixed versions
% Files to consolidate
consolidate_files = {
    'load_and_prepare_data', 
    'perform_feature_selection',
    'run_cross_validation'
};

for i = 1:length(consolidate_files)
    base_name = consolidate_files{i};
    fixed_file = fullfile(test_dir, [base_name '_fixed.m']);
    main_file = fullfile(pipeline_dir, [base_name '.m']);
    orig_file = fullfile(pipeline_dir, [base_name '_orig.m']);
    
    % Check if files exist
    if exist(fixed_file, 'file') && exist(main_file, 'file')
        % Move current to _orig if needed
        if ~exist(orig_file, 'file')
            copyfile(main_file, orig_file);
            fprintf('Created backup of %s as %s\n', main_file, orig_file);
        end
        
        % Copy fixed to main
        copyfile(fixed_file, main_file);
        fprintf('Updated %s with fixed version\n', main_file);
    else
        fprintf('Could not update %s - files missing\n', base_name);
    end
end

%% 3. Clean up redundant test files while keeping working versions
% First identify which version of the cross-validation script is best
fprintf('\nAnalyzing cross-validation scripts...\n');

% Keep the complete version for reference
complete_cv = fullfile(test_dir, 'run_cross_validation_complete.m');
if exist(complete_cv, 'file')
    fprintf('Keeping run_cross_validation_complete.m as reference implementation\n');
end

%% 4. Consolidate test outputs
% Create a results directory if it doesn't exist
results_dir = fullfile(test_dir, 'results', 'test_outputs');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
    fprintf('Created test outputs directory: %s\n', results_dir);
end

% Move all test output files to the results directory
output_files = dir(fullfile(test_dir, 'test_output_*.txt'));
for i = 1:length(output_files)
    src_file = fullfile(test_dir, output_files(i).name);
    dst_file = fullfile(results_dir, output_files(i).name);
    
    if ~exist(dst_file, 'file')
        copyfile(src_file, dst_file);
        fprintf('Moved %s to results directory\n', output_files(i).name);
    end
end

% Create a README file in the test directory
readme_content = sprintf(['# FTIR Pipeline Test Directory\n\n', ...
    '## Overview\n', ...
    'This directory contains test scripts and data for the meningioma FTIR classification pipeline.\n\n', ...
    '## Key Files\n', ...
    '- `run_test.m`: Main test script that uses fixed versions of problematic functions\n', ...
    '- `run_pipeline_test_debug.m`: Debug version with detailed output\n', ...
    '- `run_cross_validation_complete.m`: Working reference implementation of cross-validation\n', ...
    '- `test_config.m`: Test-specific configuration with adjusted parameters\n\n', ...
    '## Test Data\n', ...
    'A subset of real data is used for testing, with 14 training samples and 6 test samples.\n\n', ...
    '## Test Results\n', ...
    'Test output files are stored in the `results/test_outputs` directory.\n\n', ...
    '## Last Updated\n', ...
    'October 21, 2025\n']);

readme_file = fullfile(test_dir, 'README.md');
if ~exist(readme_file, 'file')
    fid = fopen(readme_file, 'w');
    fprintf(fid, '%s', readme_content);
    fclose(fid);
    fprintf('Created README.md file in test directory\n');
end

fprintf('\nCleanup complete!\n');