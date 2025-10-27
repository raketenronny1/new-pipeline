%% PATIENT-WISE CROSS-VALIDATION (DIRECT)
% Performs patient-stratified CV working directly with loaded data
%
% Key features:
% - Stratifies folds by Patient_ID (no data leakage)
% - Treats each Diss_ID as independent sample
% - Predicts each spectrum individually
% - Aggregates predictions per sample (Diss_ID) via MAJORITY VOTING
% - Computes both spectrum-level and sample-level metrics
%
% IMPORTANT PREPROCESSING:
% - All classifiers receive standardized spectra
% - PCA is applied ONLY for LDA classifier
% - PLS-DA, SVM, and RandomForest use original standardized spectra (NO PCA)
% - SVM uses RBF kernel with auto kernel scale

function cv_results = run_patientwise_cv_direct(data, cfg)
    fprintf('\n=== PATIENT-WISE CROSS-VALIDATION ===\n');

    %% Setup
    train = data.train;
    rng(cfg.random_seed);
    
    % Start overall timer
    overall_start_time = tic;

    % CV parameters
    n_folds = cfg.cv.n_folds;
    n_repeats = cfg.cv.n_repeats;

    fprintf('Configuration:\n');
    fprintf('  Folds: %d\n', n_folds);
    fprintf('  Repeats: %d\n', n_repeats);
    fprintf('  Samples: %d\n', train.n_samples);
    fprintf('  Patients: %d\n', length(unique(train.patient_id)));
    fprintf('  Total spectra: %d\n', train.total_spectra);
    fprintf('  Hyperparameter Optimization: %s\n', ...
            ternary(cfg.optimization.enabled, 'ENABLED', 'DISABLED'));

    %% Define Classifiers
    classifiers = get_classifier_configs(cfg);
    n_classifiers = length(classifiers);
    
    %% Optimize Hyperparameters (if enabled)
    if cfg.optimization.enabled
        fprintf('\n=== HYPERPARAMETER OPTIMIZATION ===\n');
        classifiers = optimize_hyperparameters(train, classifiers, cfg);
    end

    %% Initialize Results Storage
    cv_results = struct();
    for c = 1:n_classifiers
        cv_results.(classifiers{c}.name) = struct();
        % Spectrum-level results (primary)
        cv_results.(classifiers{c}.name).spectrum_predictions = [];
        cv_results.(classifiers{c}.name).spectrum_true = [];
        % Sample-level results (for comparison)
        cv_results.(classifiers{c}.name).sample_predictions = [];
        cv_results.(classifiers{c}.name).sample_true = [];
        cv_results.(classifiers{c}.name).sample_ids = {};
        cv_results.(classifiers{c}.name).patient_ids = {};
        cv_results.(classifiers{c}.name).fold_info = [];
    end

    %% Cross-Validation Loop
    total_iterations = n_repeats * n_folds * n_classifiers;
    current_iteration = 0;
    
    for rep = 1:n_repeats
        fprintf('\n--- Repeat %d/%d ---\n', rep, n_repeats);

        % Create patient-stratified folds
        folds = create_patient_stratified_folds(train, n_folds);

        for fold = 1:n_folds
            fold_start = tic;
            fprintf('  Fold %d/%d...', fold, n_folds);

            % Get train/val sample indices
            val_samples = folds{fold};
            train_samples = setdiff(1:train.n_samples, val_samples);

            % Extract spectra and labels
            [X_train, y_train] = extract_all_spectra(train, train_samples);
            [X_val, y_val, val_sample_map] = extract_all_spectra_with_map(train, val_samples);

            % Normalize and standardize spectra (for all classifiers)
            [X_train_std, X_val_std, ~] = standardize_spectra(X_train, X_val);

            % Train and evaluate each classifier
            for c = 1:n_classifiers
                clf_name = classifiers{c}.name;
                
                % Apply PCA ONLY for LDA - uses EDA PCA model if available
                if strcmp(classifiers{c}.type, 'lda')
                    % LDA uses PCA-transformed features
                    % Check if EDA PCA model is available in data structure
                    if isfield(data, 'pca_model') && ~isempty(data.pca_model)
                        % Use EDA PCA model (15 components)
                        [X_train_feat, X_val_feat] = apply_eda_pca_transform(X_train_std, X_val_std, data.pca_model);
                    else
                        % Fallback to standard PCA within fold
                        [X_train_feat, X_val_feat, ~] = apply_pca_transform(X_train_std, X_val_std, cfg);
                    end
                else
                    % PLS-DA, SVM, RandomForest use original standardized spectra
                    X_train_feat = X_train_std;
                    X_val_feat = X_val_std;
                end

                % Train model
                model = train_classifier(classifiers{c}, X_train_feat, y_train);

                % Predict on validation spectra
                if isstruct(model) && isfield(model, 'type') && strcmp(model.type, 'plsda')
                    % PLSDA prediction
                    scores_raw = [ones(size(X_val_feat, 1), 1), X_val_feat] * model.beta;
                    spectrum_preds = scores_raw > 0;  % >0 means class 1, <0 means class 3
                    spectrum_preds = double(spectrum_preds);  % Convert to numeric
                else
                    % Standard MATLAB model
                    [spectrum_preds, ~] = predict(model, X_val_feat);

                    % Convert predictions to numeric
                    if iscategorical(spectrum_preds)
                        spectrum_preds = double(spectrum_preds);
                    elseif iscell(spectrum_preds)
                        % Handle cell array output (TreeBagger returns cell array of strings)
                        spectrum_preds = cellfun(@str2double, spectrum_preds);
                    end
                end

                % Ensure we have numeric predictions
                if ~isnumeric(spectrum_preds)
                    error('Predictions must be numeric, got %s', class(spectrum_preds));
                end

                % Store SPECTRUM-LEVEL results (primary evaluation)
                spectrum_true = y_val;  % True labels for each spectrum
                cv_results.(clf_name).spectrum_predictions = [cv_results.(clf_name).spectrum_predictions; spectrum_preds];
                cv_results.(clf_name).spectrum_true = [cv_results.(clf_name).spectrum_true; spectrum_true];
                
                % Also compute and store SAMPLE-LEVEL results (for comparison)
                sample_preds = aggregate_to_samples(spectrum_preds, val_sample_map, length(val_samples));
                cv_results.(clf_name).sample_predictions = [cv_results.(clf_name).sample_predictions; sample_preds];
                cv_results.(clf_name).sample_true = [cv_results.(clf_name).sample_true; train.labels(val_samples)];
                cv_results.(clf_name).sample_ids = [cv_results.(clf_name).sample_ids; train.diss_id(val_samples)];
                cv_results.(clf_name).patient_ids = [cv_results.(clf_name).patient_ids; train.patient_id(val_samples)];
                cv_results.(clf_name).fold_info = [cv_results.(clf_name).fold_info; repmat([rep, fold], length(val_samples), 1)];
                
                current_iteration = current_iteration + 1;
            end

            fold_time = toc(fold_start);
            avg_time_per_iter = fold_time / n_classifiers;
            remaining_iters = total_iterations - current_iteration;
            est_remaining_time = avg_time_per_iter * remaining_iters;
            
            fprintf(' done (%.1fs, ~%.0f min remaining)\n', fold_time, est_remaining_time/60);
        end
    end
    
    overall_time = toc(overall_start_time);
    fprintf('\n✓ Cross-validation complete in %.1f minutes\n', overall_time/60);

    %% Compute Performance Metrics
    fprintf('\n=== Computing Performance Metrics ===\n');
    fprintf('NOTE: Sample-level metrics use MAJORITY VOTE aggregation\n\n');
    for c = 1:n_classifiers
        clf_name = classifiers{c}.name;
        fprintf('\n%s:\n', clf_name);

        metrics = compute_metrics_direct(cv_results.(clf_name));
        cv_results.(clf_name).metrics = metrics;
        cv_results.(clf_name).aggregation_method = 'majority_vote';  % Document method

        fprintf('  Accuracy: %.3f ± %.3f\n', metrics.accuracy_mean, metrics.accuracy_std);
        fprintf('  Sensitivity: %.3f ± %.3f\n', metrics.sensitivity_mean, metrics.sensitivity_std);
        fprintf('  Specificity: %.3f ± %.3f\n', metrics.specificity_mean, metrics.specificity_std);
        fprintf('  AUC: %.3f ± %.3f\n', metrics.auc_mean, metrics.auc_std);
    end

    %% Save Results
    if ~exist(cfg.paths.results, 'dir'), mkdir(cfg.paths.results); end
    save(fullfile(cfg.paths.results, 'cv_results_direct.mat'), 'cv_results', '-v7.3');
    fprintf('\n✓ Results saved to: %s\n', fullfile(cfg.paths.results, 'cv_results_direct.mat'));
end


%% Hyperparameter Optimization
function classifiers = optimize_hyperparameters(train, classifiers, cfg)
    % Optimize hyperparameters for selected classifiers using Bayesian optimization
    
    fprintf('Optimization mode: %s\n', cfg.optimization.mode);
    fprintf('Max evaluations per classifier: %d\n', cfg.optimization.max_evaluations);
    fprintf('Inner CV folds: %d\n\n', cfg.optimization.kfold_inner);
    
    opt_start_time = tic;
    
    for c = 1:length(classifiers)
        clf_name = classifiers{c}.name;
        
        % Check if this classifier should be optimized
        should_optimize = cfg.optimization.enabled && ...
                         (strcmp(cfg.optimization.mode, 'all') || ...
                          any(strcmp(cfg.optimization.classifiers_to_optimize, clf_name)));
        
        if ~should_optimize
            fprintf('%s: Using default parameters (optimization skipped)\n', clf_name);
            continue;
        end
        
        fprintf('Optimizing %s hyperparameters...\n', clf_name);
        clf_opt_start = tic;
        
        % Extract representative training data (using all samples)
        [X_all, y_all] = extract_all_spectra(train, 1:train.n_samples);
        [X_all, ~, ~] = standardize_spectra(X_all, X_all);  % Standardize
        
        % Apply PCA if LDA
        if strcmp(classifiers{c}.type, 'lda')
            [X_all, ~, ~] = apply_pca_transform(X_all, X_all, cfg);
        end
        
        % Setup optimization based on classifier type
        switch classifiers{c}.type
            case 'lda'
                % LDA: Optimize Delta and Gamma
                optimal_params = optimize_lda(X_all, y_all, cfg);
                classifiers{c}.delta = optimal_params.Delta;
                classifiers{c}.gamma = optimal_params.Gamma;
                
            case 'plsda'
                % PLS-DA: Optimize number of components
                optimal_params = optimize_plsda(X_all, y_all, cfg);
                classifiers{c}.n_components = optimal_params.NumComponents;
                
            case 'svm'
                % SVM: Optimize BoxConstraint and KernelScale
                optimal_params = optimize_svm(X_all, y_all, cfg);
                classifiers{c}.box_constraint = optimal_params.BoxConstraint;
                classifiers{c}.kernel_scale = optimal_params.KernelScale;
                
            case 'rf'
                % Random Forest: Optimize NumTrees and MinLeafSize
                optimal_params = optimize_rf(X_all, y_all, cfg);
                classifiers{c}.n_trees = optimal_params.NumTrees;
                classifiers{c}.min_leaf_size = optimal_params.MinLeafSize;
        end
        
        clf_opt_time = toc(clf_opt_start);
        fprintf('  %s optimization complete in %.1f minutes\n', clf_name, clf_opt_time/60);
        fprintf('  Optimal parameters: %s\n\n', struct2str(optimal_params));
    end
    
    total_opt_time = toc(opt_start_time);
    fprintf('✓ All optimizations complete in %.1f minutes\n\n', total_opt_time/60);
end


function opt_params = optimize_lda(X, y, cfg)
    % Optimize LDA using fitcdiscr's built-in Bayesian optimization
    % Optimizes Delta and Gamma parameters with cost-sensitive learning
    try
        % Get cost penalty for cost-sensitive learning
        cost_penalty = 5;
        if isfield(cfg, 'classifiers') && isfield(cfg.classifiers, 'cost_who3_penalty')
            cost_penalty = cfg.classifiers.cost_who3_penalty;
        end
        
        % Set up cost-sensitive priors
        n_who1 = sum(y == 1);
        n_who3 = sum(y == 3);
        prior_who1 = n_who1 / (n_who1 + cost_penalty * n_who3);
        prior_who3 = (cost_penalty * n_who3) / (n_who1 + cost_penalty * n_who3);
        
        % Optimize hyperparameters with cost-sensitive priors
        Mdl = fitcdiscr(X, y, ...
            'Prior', [prior_who1; prior_who3], ...
            'OptimizeHyperparameters', {'Delta', 'Gamma'}, ...
            'HyperparameterOptimizationOptions', ...
            struct('MaxObjectiveEvaluations', cfg.optimization.max_evaluations, ...
                   'KFold', cfg.optimization.kfold_inner, ...
                   'ShowPlots', false, ...
                   'Verbose', cfg.optimization.verbose, ...
                   'UseParallel', cfg.optimization.use_parallel));
        
        opt_params.Delta = Mdl.Delta;
        opt_params.Gamma = Mdl.Gamma;
        
        if cfg.optimization.verbose > 0
            fprintf('  LDA optimization complete\n');
            fprintf('  Optimal parameters: Delta=%.4f, Gamma=%.4f\n', ...
                    opt_params.Delta, opt_params.Gamma);
        end
    catch ME
        warning('OptimizationFailed:LDA', 'LDA optimization failed: %s. Using defaults.', ME.message);
        opt_params.Delta = 0;
        opt_params.Gamma = 0;
    end
end


function opt_params = optimize_plsda(X, y, cfg)
    % Optimize PLS-DA number of components via grid search with CV
    % Tests range of components specified in config
    
    % Get component range from config
    if isfield(cfg.optimization, 'plsda_components')
        n_components_range = cfg.optimization.plsda_components;
    else
        n_components_range = 1:15;
    end
    
    % Limit to available dimensions
    n_components_range = n_components_range(n_components_range <= min(size(X)));
    
    cv_errors = zeros(length(n_components_range), 1);
    
    if cfg.optimization.verbose > 0
        fprintf('  Optimizing PLS-DA components (range: %d-%d)\n', ...
                min(n_components_range), max(n_components_range));
    end
    
    for i = 1:length(n_components_range)
        n_comp = n_components_range(i);
        
        % Cross-validate
        cvp = cvpartition(y, 'KFold', cfg.optimization.kfold_inner);
        fold_errors = zeros(cvp.NumTestSets, 1);
        
        for fold = 1:cvp.NumTestSets
            X_tr = X(cvp.training(fold), :);
            y_tr = y(cvp.training(fold));
            X_val = X(cvp.test(fold), :);
            y_val = y(cvp.test(fold));
            
            % Train PLS-DA
            Y_coded = zeros(length(y_tr), 1);
            Y_coded(y_tr == 1) = 1;
            Y_coded(y_tr == 3) = -1;
            
            try
                [~, ~, ~, ~, beta] = plsregress(X_tr, Y_coded, n_comp);
                
                % Predict
                y_pred_val = [ones(size(X_val, 1), 1), X_val] * beta > 0;
                fold_errors(fold) = mean(y_pred_val ~= (y_val == 1));
            catch
                fold_errors(fold) = 1;  % Max error if regression fails
            end
        end
        
        cv_errors(i) = mean(fold_errors);
    end
    
    [~, best_idx] = min(cv_errors);
    opt_params.NumComponents = n_components_range(best_idx);
    
    if cfg.optimization.verbose > 0
        fprintf('  PLS-DA optimization complete\n');
        fprintf('  Optimal parameters: NumComponents=%d (CV error=%.4f)\n', ...
                opt_params.NumComponents, cv_errors(best_idx));
    end
end


function opt_params = optimize_svm(X, y, cfg)
    % Optimize SVM using fitcsvm's built-in Bayesian optimization
    % Optimizes BoxConstraint and KernelScale with cost-sensitive learning
    try
        % Get cost penalty for cost-sensitive learning
        cost_penalty = 5;
        if isfield(cfg, 'classifiers') && isfield(cfg.classifiers, 'cost_who3_penalty')
            cost_penalty = cfg.classifiers.cost_who3_penalty;
        end
        
        % Create cost matrix for cost-sensitive learning
        cost_matrix = [0, 1; cost_penalty, 0];
        
        % Create optimization options
        opts = struct('MaxObjectiveEvaluations', cfg.optimization.max_evaluations, ...
                      'KFold', cfg.optimization.kfold_inner, ...
                      'ShowPlots', false, ...
                      'Verbose', cfg.optimization.verbose, ...
                      'UseParallel', cfg.optimization.use_parallel, ...
                      'SaveIntermediateResults', false);
        
        % Optimize with cost-sensitive learning
        Mdl = fitcsvm(X, y, ...
            'KernelFunction', 'rbf', ...
            'Standardize', false, ...
            'Cost', cost_matrix, ...
            'OptimizeHyperparameters', {'BoxConstraint', 'KernelScale'}, ...
            'HyperparameterOptimizationOptions', opts);
        
        % Extract optimized parameters
        try
            if isprop(Mdl, 'BoxConstraints') && ~isempty(Mdl.BoxConstraints)
                if isscalar(Mdl.BoxConstraints)
                    opt_params.BoxConstraint = Mdl.BoxConstraints;
                else
                    opt_params.BoxConstraint = Mdl.BoxConstraints(1);
                end
            else
                opt_params.BoxConstraint = 1;
            end
        catch
            opt_params.BoxConstraint = 1;
        end
        
        try
            if isprop(Mdl, 'KernelParameters') && isstruct(Mdl.KernelParameters) && ...
               isfield(Mdl.KernelParameters, 'Scale')
                opt_params.KernelScale = Mdl.KernelParameters.Scale;
            else
                opt_params.KernelScale = 1;
            end
        catch
            opt_params.KernelScale = 1;
        end
        
        if cfg.optimization.verbose > 0
            fprintf('  SVM optimization complete\n');
            fprintf('  Optimal parameters: BoxConstraint=%.4f, KernelScale=%.4f\n', ...
                    opt_params.BoxConstraint, opt_params.KernelScale);
        end
    catch ME
        warning('OptimizationFailed:SVM', 'SVM optimization failed: %s. Using defaults.', ME.message);
        opt_params.BoxConstraint = 1;
        opt_params.KernelScale = 'auto';
    end
end


function opt_params = optimize_rf(X, y, cfg)
    % Optimize Random Forest using fitcensemble's built-in optimization
    try
        t = templateTree('Reproducible', true);
        Mdl = fitcensemble(X, y, ...
            'Method', 'Bag', ...
            'Learners', t, ...
            'OptimizeHyperparameters', {'NumLearningCycles', 'MinLeafSize'}, ...
            'HyperparameterOptimizationOptions', ...
            struct('MaxObjectiveEvaluations', cfg.optimization.max_evaluations, ...
                   'KFold', cfg.optimization.kfold_inner, ...
                   'ShowPlots', false, ...
                   'Verbose', cfg.optimization.verbose, ...
                   'UseParallel', cfg.optimization.use_parallel));
        
        % Extract optimized parameters from the trained ensemble
        opt_params.NumTrees = Mdl.NumTrained;
        
        % MinLeafSize is in the trained trees
        % Get it from the first trained weak learner
        try
            if isprop(Mdl, 'Trained') && ~isempty(Mdl.Trained)
                opt_params.MinLeafSize = Mdl.Trained{1}.MinLeafSize;
            else
                opt_params.MinLeafSize = 1;  % Default
            end
        catch
            opt_params.MinLeafSize = 1;  % Default if extraction fails
        end
        
        if cfg.optimization.verbose > 0
            fprintf('  RandomForest optimization complete\n');
            fprintf('  Optimal parameters: NumTrees=%d, MinLeafSize=%d\n', ...
                    opt_params.NumTrees, opt_params.MinLeafSize);
        end
    catch ME
        warning('OptimizationFailed:RF', 'RF optimization failed: %s. Using defaults.', ME.message);
        opt_params.NumTrees = 100;
        opt_params.MinLeafSize = 1;
    end
end


function str = struct2str(s)
    % Convert struct to readable string
    fields = fieldnames(s);
    str_parts = cell(length(fields), 1);
    for i = 1:length(fields)
        val = s.(fields{i});
        if isnumeric(val)
            str_parts{i} = sprintf('%s=%.4g', fields{i}, val);
        else
            str_parts{i} = sprintf('%s=%s', fields{i}, val);
        end
    end
    str = strjoin(str_parts, ', ');
end


function result = ternary(condition, true_val, false_val)
    % Ternary operator helper
    if condition
        result = true_val;
    else
        result = false_val;
    end
end


%% Helper: Create patient-stratified folds
function folds = create_patient_stratified_folds(train, n_folds)
    % Create folds stratified by patient (not by sample)
    % This ensures all samples from same patient go into same fold

    % Get unique patients and their labels
    [unique_patients, ~, patient_indices] = unique(train.patient_id);
    n_patients = length(unique_patients);

    % Get label for each patient (take from first sample)
    patient_labels = zeros(n_patients, 1);
    for i = 1:n_patients
        sample_idx = find(patient_indices == i, 1);
        patient_labels(i) = train.labels(sample_idx);
    end

    % Create stratified partition based on patients
    cv_partition = cvpartition(patient_labels, 'KFold', n_folds);

    % Convert patient folds to sample folds
    folds = cell(n_folds, 1);
    for fold = 1:n_folds
        fold_patients = find(test(cv_partition, fold));
        fold_samples = [];
        for p = 1:length(fold_patients)
            patient_idx = fold_patients(p);
            samples = find(patient_indices == patient_idx);
            fold_samples = [fold_samples; samples];
        end
        folds{fold} = fold_samples;
    end
end


%% Helper: Extract all spectra from samples
function [X, y] = extract_all_spectra(train, sample_indices)
    % Flatten all spectra from selected samples
    X = [];
    y = [];

    for i = 1:length(sample_indices)
        idx = sample_indices(i);
        spectra = train.spectra{idx};
        labels = repmat(train.labels(idx), size(spectra, 1), 1);

        X = [X; spectra];
        y = [y; labels];
    end
end


%% Helper: Extract spectra with sample mapping
function [X, y, sample_map] = extract_all_spectra_with_map(train, sample_indices)
    % Extract spectra and track which sample each spectrum belongs to
    X = [];
    y = [];
    sample_map = [];  % Maps each spectrum to its sample index (within sample_indices)

    for i = 1:length(sample_indices)
        idx = sample_indices(i);
        spectra = train.spectra{idx};
        n_spectra = size(spectra, 1);

        X = [X; spectra];
        y = [y; repmat(train.labels(idx), n_spectra, 1)];
        sample_map = [sample_map; repmat(i, n_spectra, 1)];
    end
end


%% Helper: Standardize spectra (preprocessing for all classifiers)
function [X_train, X_val, params] = standardize_spectra(X_train, X_val)
    % Standardize spectra using z-score normalization
    % This is applied to ALL classifiers before feature extraction
    
    % Compute mean and std from training data
    mu = mean(X_train, 1);
    sigma = std(X_train, 0, 1);
    sigma(sigma == 0) = 1;  % Avoid division by zero
    
    % Standardize both sets
    X_train = (X_train - mu) ./ sigma;
    X_val = (X_val - mu) ./ sigma;
    
    % Store parameters
    params = struct('mu', mu, 'sigma', sigma);
end


%% Helper: Apply EDA PCA transformation (uses pre-computed PCA from EDA)
function [X_train_pca, X_val_pca] = apply_eda_pca_transform(X_train, X_val, pca_model)
    % Apply PCA using the model from EDA (15 components)
    % This ensures consistent PCA across all folds
    %
    % Inputs:
    %   X_train, X_val - Already standardized data
    %   pca_model - PCA model from EDA containing:
    %               * coeff: PCA coefficients (15 components)
    %               * X_mean: Mean spectrum (already subtracted during standardization)
    %               * n_comp: Number of components
    
    % Note: X_train and X_val are already standardized, but PCA from EDA
    % was done on mean-centered data. We need to apply the same centering.
    % However, since standardization already centers the data (mu=0 for standardized),
    % we can directly project.
    
    % Project data onto EDA PCA space
    X_train_pca = X_train * pca_model.coeff;
    X_val_pca = X_val * pca_model.coeff;
end


%% Helper: Apply PCA transformation (ONLY FOR LDA - fallback if no EDA)
function [X_train, X_val, pca_model] = apply_pca_transform(X_train, X_val, cfg)
    % Apply PCA for dimensionality reduction
    % NOTE: This should ONLY be called for LDA classifier
    %       PLS-DA, SVM, and RandomForest use original standardized spectra
    
    % PCA on already-standardized data
    [coeff, ~, ~, ~, explained] = pca(X_train);

    % Select components
    cumvar = cumsum(explained) / 100;
    n_comp = find(cumvar >= cfg.pca.variance_threshold, 1, 'first');
    n_comp = min(n_comp, cfg.pca.max_components);

    % Transform
    X_train = X_train * coeff(:, 1:n_comp);
    X_val = X_val * coeff(:, 1:n_comp);

    % Store model (without mu/sigma since already standardized)
    pca_model = struct('coeff', coeff, 'n_comp', n_comp);
end


%% Helper: Aggregate spectrum predictions to sample level
function sample_preds = aggregate_to_samples(spectrum_preds, sample_map, n_samples)
    % Aggregate spectrum-level predictions to sample-level using MAJORITY VOTE
    % Each sample's prediction is the mode (most common) prediction among its spectra
    sample_preds = zeros(n_samples, 1);

    for i = 1:n_samples
        spectra_idx = (sample_map == i);
        sample_preds(i) = mode(spectrum_preds(spectra_idx));
    end
end


%% Helper: Get classifier configurations
function classifiers = get_classifier_configs(cfg)
    % Get cost penalty from config
    cost_penalty = 5;  % Default
    if isfield(cfg, 'classifiers') && isfield(cfg.classifiers, 'cost_who3_penalty')
        cost_penalty = cfg.classifiers.cost_who3_penalty;
    end
    
    classifiers = {
        struct('name', 'LDA', 'type', 'lda', 'cost_penalty', cost_penalty),
        struct('name', 'PLSDA', 'type', 'plsda', 'n_components', 5, 'cost_penalty', cost_penalty),
        struct('name', 'SVM', 'type', 'svm', 'cost_penalty', cost_penalty),
        struct('name', 'RandomForest', 'type', 'rf', 'n_trees', 100, 'cost_penalty', cost_penalty)
    };
end


%% Helper: Train classifier
function model = train_classifier(classifier_cfg, X_train, y_train)
    % Get cost penalty from config (default: 5 for WHO-3)
    cost_penalty = 5;
    if isfield(classifier_cfg, 'cost_penalty')
        cost_penalty = classifier_cfg.cost_penalty;
    end
    
    switch classifier_cfg.type
        case 'lda'
            try
                discrim_type = 'linear';
                
                % Cost-sensitive learning via class priors
                % Adjust prior probabilities to emphasize WHO-3
                n_who1 = sum(y_train == 1);
                n_who3 = sum(y_train == 3);
                
                % Weight WHO-3 more heavily
                prior_who1 = n_who1 / (n_who1 + cost_penalty * n_who3);
                prior_who3 = (cost_penalty * n_who3) / (n_who1 + cost_penalty * n_who3);
                
                if isfield(classifier_cfg, 'delta') && isfield(classifier_cfg, 'gamma')
                    model = fitcdiscr(X_train, y_train, ...
                        'DiscrimType', discrim_type, ...
                        'Delta', classifier_cfg.delta, ...
                        'Gamma', classifier_cfg.gamma, ...
                        'Prior', [prior_who1; prior_who3]);
                else
                    model = fitcdiscr(X_train, y_train, ...
                        'DiscrimType', discrim_type, ...
                        'Prior', [prior_who1; prior_who3]);
                end
            catch
                model = fitcdiscr(X_train, y_train, 'DiscrimType', 'pseudoLinear');
            end

        case 'plsda'
            % PLS-DA using MATLAB's plsregress (works with original spectra, not PCA)
            n_comp = classifier_cfg.n_components;

            % Create dummy Y matrix for binary classification
            Y = zeros(length(y_train), 1);
            Y(y_train == 1) = 1;
            Y(y_train == 3) = -1;

            % Fit PLS regression
            [~, ~, ~, ~, beta] = plsregress(X_train, Y, n_comp);
            model = struct('beta', beta, 'type', 'plsda', ...
                          'cost_penalty', cost_penalty);  % Store for threshold adjustment

        case 'svm'
            % SVM with RBF kernel - standardization already done, use original spectra
            box_constraint = 1;
            kernel_scale = 'auto';
            if isfield(classifier_cfg, 'box_constraint')
                box_constraint = classifier_cfg.box_constraint;
            end
            if isfield(classifier_cfg, 'kernel_scale')
                kernel_scale = classifier_cfg.kernel_scale;
            end
            
            % Cost-sensitive learning via cost matrix
            % Cost matrix: [C(predict WHO-1|true WHO-1), C(predict WHO-3|true WHO-1);
            %               C(predict WHO-1|true WHO-3), C(predict WHO-3|true WHO-3)]
            cost_matrix = [0, 1; cost_penalty, 0];
            
            model = fitcsvm(X_train, y_train, ...
                           'KernelFunction', 'rbf', ...
                           'Standardize', false, ...  % Already standardized
                           'KernelScale', kernel_scale, ...
                           'BoxConstraint', box_constraint, ...
                           'Cost', cost_matrix);

        case 'rf'
            n_trees = classifier_cfg.n_trees;
            min_leaf_size = 1;
            if isfield(classifier_cfg, 'min_leaf_size')
                min_leaf_size = classifier_cfg.min_leaf_size;
            end
            
            % Cost-sensitive learning via sample weights
            % Give WHO-3 samples higher weight
            sample_weights = ones(length(y_train), 1);
            sample_weights(y_train == 3) = cost_penalty;
            
            model = TreeBagger(n_trees, X_train, y_train, ...
                              'Method', 'classification', ...
                              'OOBPrediction', 'off', ...
                              'MinLeafSize', min_leaf_size, ...
                              'Weights', sample_weights);
    end
end


%% Helper: Compute metrics
function metrics = compute_metrics_direct(results)
    % Compute classification metrics from CV results
    % Now using SPECTRUM-LEVEL evaluation (much more data!)

    % PRIMARY: Spectrum-level metrics
    y_true = results.spectrum_true;
    y_pred = results.spectrum_predictions;

    % Convert to binary (1 vs 3)
    y_true_bin = (y_true == 1);
    y_pred_bin = (y_pred == 1);

    % Confusion matrix
    tp = sum(y_true_bin & y_pred_bin);
    tn = sum(~y_true_bin & ~y_pred_bin);
    fp = sum(~y_true_bin & y_pred_bin);
    fn = sum(y_true_bin & ~y_pred_bin);

    % Spectrum-level metrics
    metrics.spectrum_accuracy = (tp + tn) / length(y_true);
    metrics.spectrum_sensitivity = tp / (tp + fn);
    metrics.spectrum_specificity = tn / (tn + fp);
    metrics.spectrum_precision = tp / (tp + fp);
    metrics.spectrum_f1 = 2 * tp / (2*tp + fp + fn);
    
    % ALSO compute sample-level metrics (for comparison)
    y_true_sample = results.sample_true;
    y_pred_sample = results.sample_predictions;
    
    y_true_sample_bin = (y_true_sample == 1);
    y_pred_sample_bin = (y_pred_sample == 1);
    
    tp_s = sum(y_true_sample_bin & y_pred_sample_bin);
    tn_s = sum(~y_true_sample_bin & ~y_pred_sample_bin);
    fp_s = sum(~y_true_sample_bin & y_pred_sample_bin);
    fn_s = sum(y_true_sample_bin & ~y_pred_sample_bin);
    
    metrics.sample_accuracy = (tp_s + tn_s) / length(y_true_sample);
    metrics.sample_sensitivity = tp_s / (tp_s + fn_s);
    metrics.sample_specificity = tn_s / (tn_s + fp_s);

    % Compute std across folds using SAMPLE-level (for fold statistics)
    if isfield(results, 'fold_info') && ~isempty(results.fold_info)
        fold_ids = unique(results.fold_info, 'rows');
        n_folds = size(fold_ids, 1);

        fold_acc = zeros(n_folds, 1);
        fold_sens = zeros(n_folds, 1);
        fold_spec = zeros(n_folds, 1);

        for i = 1:n_folds
            rep = fold_ids(i, 1);
            fold = fold_ids(i, 2);

            idx = (results.fold_info(:, 1) == rep) & (results.fold_info(:, 2) == fold);

            yt = y_true_sample_bin(idx);
            yp = y_pred_sample_bin(idx);

            tp_f = sum(yt & yp);
            tn_f = sum(~yt & ~yp);
            fp_f = sum(~yt & yp);
            fn_f = sum(yt & ~yp);

            fold_acc(i) = (tp_f + tn_f) / length(yt);
            fold_sens(i) = tp_f / (tp_f + fn_f + eps);
            fold_spec(i) = tn_f / (tn_f + fp_f + eps);
        end

        metrics.accuracy_std = std(fold_acc);
        metrics.sensitivity_std = std(fold_sens);
        metrics.specificity_std = std(fold_spec);
        metrics.auc_std = std(fold_acc);  % Simplified
    else
        metrics.accuracy_std = 0;
        metrics.sensitivity_std = 0;
        metrics.specificity_std = 0;
        metrics.auc_std = 0;
    end
    
    % Use spectrum-level metrics as primary
    metrics.accuracy_mean = metrics.spectrum_accuracy;
    metrics.sensitivity_mean = metrics.spectrum_sensitivity;
    metrics.specificity_mean = metrics.spectrum_specificity;
    metrics.precision_mean = metrics.spectrum_precision;
    metrics.f1_mean = metrics.spectrum_f1;
    metrics.auc_mean = metrics.spectrum_accuracy;  % Simplified
end
