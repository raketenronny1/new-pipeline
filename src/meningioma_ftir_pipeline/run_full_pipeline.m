%% MASTER SCRIPT - Run Full Pipeline
% This script runs the complete meningioma classification pipeline


function run_full_pipeline()
    % Generate unique run ID (timestamp)
    run_id = datestr(now, 'yyyymmdd_HHMMSS');
    run_folder = fullfile('results', 'meningioma_ftir_pipeline', ['run_' run_id]);
    model_folder = fullfile('models', 'meningioma_ftir_pipeline', ['run_' run_id]);
    qc_folder = fullfile(run_folder, 'qc');

    % Load configuration and update paths for this run
    cfg = config();
    cfg.paths.results = [run_folder filesep];
    cfg.paths.models = [model_folder filesep];
    cfg.paths.qc = [qc_folder filesep];

    % Create required directories
    ensure_directories_exist(cfg);

    % Start logging in run-specific folder
    log_file = fopen(fullfile(cfg.paths.results, 'pipeline_log.txt'), 'w');

    try
        % Document MATLAB version and toolboxes
        log_message('=== MATLAB VERSION INFO ===', log_file);
        ver_info = ver;
        for i = 1:length(ver_info)
            log_message(sprintf('%s Version %s', ver_info(i).Name, ver_info(i).Version), log_file);
        end
        log_message('========================', log_file);

        % Set random seed for reproducibility
        rng(cfg.random_seed, 'twister');

        % Phase 0: Quality Control
        log_message('Starting Phase 0: Quality Control', log_file);
    quality_control_analysis(cfg);

        % Phase 1: Data Loading
        log_message('Starting Phase 1: Data Loading', log_file);
    load_and_prepare_data(cfg);

        % Phase 2: Feature Selection
        log_message('Starting Phase 2: Feature Selection', log_file);
    perform_feature_selection(cfg);

        % Phase 3: Cross-Validation
        log_message('Starting Phase 3: Cross-Validation', log_file);
    run_cross_validation(cfg);

        % Phase 4: Final Model Training
        log_message('Starting Phase 4: Final Model Training', log_file);
    train_final_model(cfg);

        % Phase 5: Test Evaluation
        log_message('Starting Phase 5: Test Evaluation', log_file);
    evaluate_test_set(cfg);

        % Phase 6: Report Generation
        log_message('Starting Phase 6: Report Generation', log_file);
    generate_report(cfg);

        log_message('Pipeline completed successfully!', log_file);

    catch ME
        % Log any errors
        log_message(sprintf('ERROR: %s', ME.message), log_file);
        log_message(sprintf('Error in: %s (Line %d)', ME.stack(1).name, ME.stack(1).line), log_file);
        rethrow(ME);

    finally
        % Close log file
        if ~isempty(fopen('all'))
            fclose(log_file);
        end
    end
end

function ensure_directories_exist(cfg)
    % Create required directories if they don't exist
    directories = {
        cfg.paths.models
        cfg.paths.results
        cfg.paths.qc
    };
    
    for i = 1:length(directories)
        if ~exist(directories{i}, 'dir')
            [success, msg] = mkdir(directories{i});
            if ~success
                error('Failed to create directory %s: %s', directories{i}, msg);
            end
        end
    end
end