function run_test_with_debug()
    % This function will run the pipeline with detailed debug output
    try
        fprintf('Starting test with debug output\n');
        
        % Call the main test function
        run_pipeline_test_debug();
        
    catch ME
        fprintf('\n===== ERROR OCCURRED =====\n');
        fprintf('Error message: %s\n', ME.message);
        
        % Display stack trace
        for i = 1:length(ME.stack)
            fprintf('Function: %s, Line: %d\n', ME.stack(i).name, ME.stack(i).line);
        end
        
        % Check if the feature selection fixed function exists
        if exist('perform_feature_selection_fixed', 'file') == 2
            fprintf('The perform_feature_selection_fixed function exists on the path\n');
        else
            fprintf('WARNING: The perform_feature_selection_fixed function is NOT on the path\n');
            fprintf('Current directory: %s\n', pwd);
        end
        
        % List the contents of the most recent results directory
        try
            results_dir = dir(fullfile('results', 'run_*'));
            dates = [results_dir.datenum];
            [~, idx] = max(dates);
            latest_dir = fullfile('results', results_dir(idx).name);
            
            fprintf('\nLatest results directory: %s\n', latest_dir);
            dir_contents = dir(latest_dir);
            for i = 1:length(dir_contents)
                fprintf('  %s\n', dir_contents(i).name);
            end
            
            % Check if X_train_pca.mat exists and what's in it
            pca_file = fullfile(latest_dir, 'X_train_pca.mat');
            if exist(pca_file, 'file')
                fprintf('\nX_train_pca.mat exists. Contents:\n');
                fileinfo = who('-file', pca_file);
                for i = 1:length(fileinfo)
                    fprintf('  %s\n', fileinfo{i});
                end
            else
                fprintf('\nX_train_pca.mat does not exist in the latest results directory\n');
            end
        catch
            fprintf('Could not check latest results directory\n');
        end
    end
end