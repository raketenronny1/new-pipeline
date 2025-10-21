%% PHASE 3: MODEL SELECTION VIA CROSS-VALIDATION
% This script performs model selection using cross-validation on the training set

function run_cross_validation(cfg)
    % Input validation
    if ~isstruct(cfg) || ~isfield(cfg, 'paths') || ~isfield(cfg.paths, 'results')
        error('Invalid cfg structure. Must contain paths.results');
    end

    %% Load Data
    try
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

        % CV parameters
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

        % Pre-allocate memory for results
        all_predictions = cell(n_folds * n_repeats, 1);
        all_true_labels = cell(n_folds * n_repeats, 1);

        % Initialize results storage
        cv_results = cell(4, 1);  % One cell per classifier
        classifierNames = {'LDA', 'PLSDA', 'SVM', 'RandomForest'};

        %% Define Hyperparameter Grids
        % PLS-DA: Number of components
        plsda_n_comp_grid = 1:min(10, size(X_train_pca, 2));

        % SVM: C and gamma
        svm_C_grid = 10.^(-2:0.5:2);  % [0.01, 0.03, ..., 100]
        svm_gamma_grid = 10.^(-3:0.5:1);  % [0.001, 0.003, ..., 10]

        % Random Forest: n_trees and max_depth
        rf_n_trees_grid = [50, 100, 200, 500];
        rf_max_depth_grid = [5, 10, 20, 30];