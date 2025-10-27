%% RUN_PIPELINE - Unified entry point for FTIR meningioma classification pipeline
%
% This is the main entry point for running the complete machine learning pipeline
% for FTIR-based meningioma classification (WHO Grade 1 vs 3). The pipeline includes:
%   1. Exploratory Data Analysis (EDA) with outlier detection
%   2. Data loading with quality filtering
%   3. Patient-stratified cross-validation
%   4. Results export and visualization
%
% SYNTAX:
%   run_pipeline()
%   run_pipeline('Name', Value, ...)
%   results = run_pipeline(...)
%
% OPTIONAL NAME-VALUE PAIRS:
%   'RunEDA'         - Run EDA step (default: true, set false if already done)
%   'OutlierMethod'  - Outlier detection method: 'eda', 'qc', 'none' (default: 'eda')
%   'Classifiers'    - Cell array of classifiers to run (default: {'LDA','PLSDA','SVM','RandomForest'})
%   'NFolds'         - Number of CV folds (default: from config, usually 5)
%   'NRepeats'       - Number of CV repeats (default: from config, usually 50)
%   'Verbose'        - Display detailed output (default: true)
%   'SaveResults'    - Save results to disk (default: true)
%
% OUTPUTS:
%   results - Structure containing:
%             * cv_results: Cross-validation results
%             * data: Loaded data structure
%             * config: Configuration used
%             * timestamp: Run timestamp
%
% WORKFLOW:
%   Step 1: Run EDA (if requested)
%           - Loads training data
%           - Performs PCA analysis
%           - Detects outliers using T² and Q statistics
%           - Saves results to results/eda/
%
%   Step 2: Load data with quality filtering
%           - Applies EDA/QC outlier removal
%           - Packages data for classification
%           - Includes PCA model if using EDA
%
%   Step 3: Cross-validation
%           - Patient-stratified k-fold CV
%           - Tests multiple classifiers
%           - Hyperparameter optimization (if enabled)
%
%   Step 4: Export results
%           - Creates Excel files with predictions
%           - Generates text summaries
%           - Saves MATLAB result structures
%
% EXAMPLES:
%   % Run complete pipeline with defaults (recommended)
%   run_pipeline()
%
%   % Run pipeline without EDA (assumes EDA already done)
%   run_pipeline('RunEDA', false)
%
%   % Quick test with reduced CV
%   run_pipeline('NFolds', 3, 'NRepeats', 10)
%
%   % Test only SVM classifier
%   run_pipeline('Classifiers', {'SVM'})
%
%   % Use legacy QC instead of EDA
%   run_pipeline('OutlierMethod', 'qc', 'RunEDA', false)
%
%   % Capture results for further analysis
%   results = run_pipeline();
%   best_clf = find_best_classifier(results.cv_results);
%
% NOTES:
%   - First run should use RunEDA=true (default)
%   - Subsequent runs can set RunEDA=false to save time
%   - Results are saved in results/meningioma_ftir_pipeline/run_TIMESTAMP/
%   - EDA results are saved in results/eda/
%   - Requires data files: data_table_train.mat, data_table_test.mat, wavenumbers.mat
%
% DEPENDENCIES:
%   - config.m
%   - run_eda.m
%   - load_pipeline_data.m
%   - run_patientwise_cv_direct.m
%   - export_cv_results.m (in src/utils/)
%
% See also: config, run_eda, load_pipeline_data, run_patientwise_cv_direct

function results = run_pipeline(varargin)
    %% Parse input arguments
    p = inputParser;
    addParameter(p, 'RunEDA', true, @islogical);
    addParameter(p, 'OutlierMethod', 'eda', @(x) ismember(x, {'eda', 'qc', 'none'}));
    addParameter(p, 'Classifiers', {'LDA', 'PLSDA', 'SVM', 'RandomForest'}, @iscell);
    addParameter(p, 'NFolds', [], @(x) isempty(x) || (isnumeric(x) && x > 0));
    addParameter(p, 'NRepeats', [], @(x) isempty(x) || (isnumeric(x) && x > 0));
    addParameter(p, 'Verbose', true, @islogical);
    addParameter(p, 'SaveResults', true, @islogical);
    parse(p, varargin{:});
    
    opts = p.Results;
    
    %% Load configuration
    fprintf('\n╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║  MENINGIOMA FTIR CLASSIFICATION PIPELINE                  ║\n');
    fprintf('║  Unified EDA-based Machine Learning Pipeline              ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n\n');
    
    cfg = config();
    
    % Override CV parameters if provided
    if ~isempty(opts.NFolds)
        cfg.cv.n_folds = opts.NFolds;
    end
    if ~isempty(opts.NRepeats)
        cfg.cv.n_repeats = opts.NRepeats;
    end
    
    % Set random seed for reproducibility
    rng(cfg.random_seed, 'twister');
    
    % Create timestamped results directory
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    run_dir = fullfile(cfg.paths.results, ['run_' timestamp]);
    if ~exist(run_dir, 'dir')
        mkdir(run_dir);
    end
    cfg.paths.results = [run_dir filesep];
    
    if opts.Verbose
        fprintf('Configuration:\n');
        fprintf('  Outlier method: %s\n', opts.OutlierMethod);
        fprintf('  Classifiers: %s\n', strjoin(opts.Classifiers, ', '));
        fprintf('  CV: %d folds × %d repeats\n', cfg.cv.n_folds, cfg.cv.n_repeats);
        fprintf('  Random seed: %d\n', cfg.random_seed);
        fprintf('  Results directory: %s\n\n', run_dir);
    end
    
    %% Step 1: Run EDA (if requested)
    if opts.RunEDA
        fprintf('═══════════════════════════════════════════════════════════\n');
        fprintf(' STEP 1: EXPLORATORY DATA ANALYSIS\n');
        fprintf('═══════════════════════════════════════════════════════════\n');
        
        run_eda('Verbose', opts.Verbose);
        
        fprintf('\n✓ EDA completed\n\n');
    else
        if opts.Verbose
            fprintf('Skipping EDA (using existing results)\n\n');
        end
    end
    
    %% Step 2: Load data with quality filtering
    fprintf('═══════════════════════════════════════════════════════════\n');
    fprintf(' STEP 2: DATA LOADING & QUALITY FILTERING\n');
    fprintf('═══════════════════════════════════════════════════════════\n');
    
    % Add src/utils to path for export functions
    addpath(fullfile(pwd, 'src', 'utils'));
    
    data = load_pipeline_data(cfg, ...
                              'OutlierMethod', opts.OutlierMethod, ...
                              'Verbose', opts.Verbose);
    
    fprintf('\n✓ Data loading completed\n\n');
    
    %% Step 3: Cross-validation
    fprintf('═══════════════════════════════════════════════════════════\n');
    fprintf(' STEP 3: PATIENT-STRATIFIED CROSS-VALIDATION\n');
    fprintf('═══════════════════════════════════════════════════════════\n');
    
    % Filter classifiers based on user selection
    original_classifiers = cfg.classifiers.types;
    cfg.classifiers.types = opts.Classifiers;
    
    if opts.Verbose
        fprintf('Running CV with %d folds, %d repeats...\n', ...
                cfg.cv.n_folds, cfg.cv.n_repeats);
        fprintf('This may take several minutes...\n\n');
    end
    
    tic;
    cv_results = run_patientwise_cv_direct(data, cfg);
    elapsed = toc;
    
    if opts.Verbose
        fprintf('\n✓ Cross-validation completed in %.1f minutes\n\n', elapsed/60);
    end
    
    %% Step 4: Export results
    if opts.SaveResults
        fprintf('═══════════════════════════════════════════════════════════\n');
        fprintf(' STEP 4: EXPORTING RESULTS\n');
        fprintf('═══════════════════════════════════════════════════════════\n');
        
        % Determine pipeline description
        switch opts.OutlierMethod
            case 'eda'
                pipeline_desc = 'EDA-based outlier removal (T²-Q detection)';
            case 'qc'
                pipeline_desc = 'Legacy QC filtering';
            case 'none'
                pipeline_desc = 'No outlier filtering';
        end
        
        % Export CV results
        export_cv_results(cv_results, cfg.paths.results, ...
                         'Pipeline', pipeline_desc, ...
                         'Verbose', opts.Verbose);
        
        % Save MATLAB results
        results_file = fullfile(cfg.paths.results, 'cv_results.mat');
        save(results_file, 'cv_results', 'data', 'cfg', 'timestamp');
        
        if opts.Verbose
            fprintf('  MATLAB results saved: cv_results.mat\n');
            fprintf('\n✓ Results export completed\n\n');
        end
    end
    
    %% Display summary
    fprintf('═══════════════════════════════════════════════════════════\n');
    fprintf(' PIPELINE SUMMARY\n');
    fprintf('═══════════════════════════════════════════════════════════\n');
    
    % Get classifier names
    classifier_names = fieldnames(cv_results);
    classifier_names = classifier_names(~strcmp(classifier_names, 'metadata'));
    
    fprintf('Performance (mean ± std):\n\n');
    fprintf('%-15s %10s %14s %14s %10s\n', ...
            'Classifier', 'Accuracy', 'Sensitivity', 'Specificity', 'AUC');
    fprintf('%-15s %10s %14s %14s %10s\n', ...
            '----------', '--------', '-----------', '-----------', '---');
    
    for i = 1:length(classifier_names)
        clf_name = classifier_names{i};
        m = cv_results.(clf_name).metrics;
        
        fprintf('%-15s %.3f±%.3f   %.3f±%.3f    %.3f±%.3f   %.3f±%.3f\n', ...
                clf_name, ...
                m.accuracy_mean, m.accuracy_std, ...
                m.sensitivity_mean, m.sensitivity_std, ...
                m.specificity_mean, m.specificity_std, ...
                m.auc_mean, m.auc_std);
    end
    
    fprintf('\n');
    fprintf('Results saved to: %s\n', run_dir);
    fprintf('\n╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║  PIPELINE COMPLETED SUCCESSFULLY                          ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n\n');
    
    %% Package output
    if nargout > 0
        results = struct();
        results.cv_results = cv_results;
        results.data = data;
        results.config = cfg;
        results.timestamp = timestamp;
        results.elapsed_time_minutes = elapsed / 60;
    end
    
    % Restore original classifier list
    cfg.classifiers.types = original_classifiers;
end
