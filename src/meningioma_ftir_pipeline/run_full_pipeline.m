%% MASTER SCRIPT - Run Full Pipeline
% This script runs the complete meningioma classification pipeline


function run_full_pipeline(cfg)
    % Input validation
    if nargin < 1
        % If no config provided, load default
        cfg = config();
    elseif ~isstruct(cfg) || ~isfield(cfg, 'paths')
        error('Invalid cfg structure. Must contain paths field.');
    end
    
    % Generate unique run ID (timestamp)
    run_id = datestr(now, 'yyyymmdd_HHMMSS');
    run_folder = fullfile(cfg.paths.results, ['run_' run_id]);
    model_folder = fullfile(cfg.paths.models, ['run_' run_id]);
    qc_folder = fullfile(run_folder, 'qc');

    % Update paths for this run
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
        if isfield(cfg, 'random_seed')
            rng(cfg.random_seed, 'twister');
        else
            cfg.random_seed = 42;  % Default seed
            warning('No random seed specified in cfg. Using default seed 42.');
            rng(cfg.random_seed, 'twister');
        end

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
    cv_results = run_cross_validation(cfg);

        % Phase 4: Final Model Training
        log_message('Starting Phase 4: Final Model Training', log_file);
    final_model = train_final_model(cfg, cv_results);

        % Phase 5: Test Evaluation
        log_message('Starting Phase 5: Test Evaluation', log_file);
    test_results = evaluate_test_set(cfg, final_model);

        % Phase 6: Report Generation
        log_message('Starting Phase 6: Report Generation', log_file);
    generate_report(cfg, cv_results, final_model, test_results);

        log_message('Pipeline completed successfully!', log_file);

    catch ME
        % Log any errors and close log file
        if ~isempty(log_file)
            log_message(sprintf('ERROR: %s', ME.message), log_file);
            log_message(sprintf('Error in: %s (Line %d)', ME.stack(1).name, ME.stack(1).line), log_file);
            fclose(log_file);
        end
        rethrow(ME);
    end
    
    % Close log file if no errors occurred
    if ~isempty(log_file)
        fclose(log_file);
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