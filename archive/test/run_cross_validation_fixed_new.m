%% PHASE 3: MODEL SELECTION VIA CROSS-VALIDATION
% This script performs model selection using cross-validation on the training set

function run_cross_validation(cfg)
    % Input validation
    if ~isstruct(cfg) || ~isfield(cfg, 'paths') || ~isfield(cfg.paths, 'results')
        error('Invalid cfg structure. Must contain paths.results');
    end

    %% Load Data
    fprintf('Loading transformed training data...\n');
    load(fullfile(cfg.paths.results, 'X_train_pca.mat'), 'X_train_pca');
    load(fullfile(cfg.paths.results, 'preprocessed_data.mat'), 'trainingData');

    %% Set Up Cross-Validation
    fprintf('Setting up cross-validation...\n');

    % Set random seed from config or use default
    if isfield(cfg, 'random_seed')
        rng(cfg.random_seed, 'twister');
    else
        rng(42, 'twister');
        warning('No random seed specified in cfg. Using default seed 42.');
    end

    % CV parameters - use config values or defaults
    if isfield(cfg, 'cv') && isfield(cfg.cv, 'n_folds')
        n_folds = cfg.cv.n_folds;
        fprintf('Using %d folds from configuration\n', n_folds);
    else
        n_folds = 5;
        fprintf('Using default %d folds\n', n_folds);
    end
    
    if isfield(cfg, 'cv') && isfield(cfg.cv, 'n_repeats')
        n_repeats = cfg.cv.n_repeats;
        fprintf('Using %d repeats from configuration\n', n_repeats);
    else
        n_repeats = 50;
        fprintf('Using default %d repeats\n', n_repeats);
    end

    % Print CV settings
    fprintf('Cross-validation settings: %d folds, %d repeats\n', n_folds, n_repeats);

    % Pre-allocate memory for results
    all_predictions = cell(n_folds * n_repeats, 1);
    all_true_labels = cell(n_folds * n_repeats, 1);

    % Initialize results storage
    cv_results = cell(4, 1);  % One cell per classifier
    classifierNames = {'LDA', 'PLSDA', 'SVM', 'RandomForest'};