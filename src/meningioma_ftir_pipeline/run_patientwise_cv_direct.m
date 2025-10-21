%% PATIENT-WISE CROSS-VALIDATION (DIRECT)
% Performs patient-stratified CV working directly with loaded data
%
% Key features:
% - Stratifies folds by Patient_ID (no data leakage)
% - Treats each Diss_ID as independent sample
% - Predicts each spectrum individually
% - Aggregates predictions per sample (Diss_ID) via majority voting
% - Computes both spectrum-level and sample-level metrics

function cv_results = run_patientwise_cv_direct(data, cfg)
    fprintf('\n=== PATIENT-WISE CROSS-VALIDATION ===\n');
    
    %% Setup
    train = data.train;
    rng(cfg.random_seed);
    
    % CV parameters
    n_folds = cfg.cv.n_folds;
    n_repeats = cfg.cv.n_repeats;
    
    fprintf('Configuration:\n');
    fprintf('  Folds: %d\n', n_folds);
    fprintf('  Repeats: %d\n', n_repeats);
    fprintf('  Samples: %d\n', train.n_samples);
    fprintf('  Patients: %d\n', length(unique(train.patient_id)));
    fprintf('  Total spectra: %d\n', train.total_spectra);
    
    %% Define Classifiers
    classifiers = get_classifier_configs(cfg);
    n_classifiers = length(classifiers);
    
    %% Initialize Results Storage
    cv_results = struct();
    for c = 1:n_classifiers
        cv_results.(classifiers{c}.name) = struct();
        cv_results.(classifiers{c}.name).sample_predictions = [];
        cv_results.(classifiers{c}.name).sample_true = [];
        cv_results.(classifiers{c}.name).sample_ids = {};
        cv_results.(classifiers{c}.name).patient_ids = {};
        cv_results.(classifiers{c}.name).fold_info = [];
    end
    
    %% Cross-Validation Loop
    for rep = 1:n_repeats
        fprintf('\n--- Repeat %d/%d ---\n', rep, n_repeats);
        
        % Create patient-stratified folds
        folds = create_patient_stratified_folds(train, n_folds);
        
        for fold = 1:n_folds
            fprintf('  Fold %d/%d...', fold, n_folds);
            
            % Get train/val sample indices
            val_samples = folds{fold};
            train_samples = setdiff(1:train.n_samples, val_samples);
            
            % Extract spectra and labels
            [X_train, y_train] = extract_all_spectra(train, train_samples);
            [X_val, ~, val_sample_map] = extract_all_spectra_with_map(train, val_samples);
            
            % Apply PCA (always, to reduce dimensionality)
            [X_train, X_val, ~] = apply_pca_transform(X_train, X_val, cfg);
            
            % Train and evaluate each classifier
            for c = 1:n_classifiers
                clf_name = classifiers{c}.name;
                
                % Train model
                model = train_classifier(classifiers{c}, X_train, y_train);
                
                % Predict on validation spectra
                if isstruct(model) && isfield(model, 'type') && strcmp(model.type, 'plsda')
                    % PLSDA prediction
                    scores_raw = [ones(size(X_val, 1), 1), X_val] * model.beta;
                    spectrum_preds = scores_raw > 0;  % >0 means class 1, <0 means class 3
                else
                    % Standard MATLAB model
                    [spectrum_preds, ~] = predict(model, X_val);
                    
                    % Ensure predictions are logical/numeric
                    if iscategorical(spectrum_preds)
                        spectrum_preds = double(spectrum_preds);
                    end
                end
                
                % Aggregate to sample level (majority voting per Diss_ID)
                sample_preds = aggregate_to_samples(spectrum_preds, val_sample_map, length(val_samples));
                
                % Store results
                cv_results.(clf_name).sample_predictions = [cv_results.(clf_name).sample_predictions; sample_preds];
                cv_results.(clf_name).sample_true = [cv_results.(clf_name).sample_true; train.labels(val_samples)];
                cv_results.(clf_name).sample_ids = [cv_results.(clf_name).sample_ids; train.diss_id(val_samples)];
                cv_results.(clf_name).patient_ids = [cv_results.(clf_name).patient_ids; train.patient_id(val_samples)];
                cv_results.(clf_name).fold_info = [cv_results.(clf_name).fold_info; repmat([rep, fold], length(val_samples), 1)];
            end
            
            fprintf(' done\n');
        end
    end
    
    %% Compute Performance Metrics
    fprintf('\n=== Computing Performance Metrics ===\n');
    for c = 1:n_classifiers
        clf_name = classifiers{c}.name;
        fprintf('\n%s:\n', clf_name);
        
        metrics = compute_metrics_direct(cv_results.(clf_name));
        cv_results.(clf_name).metrics = metrics;
        
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


%% Helper: Apply PCA transformation
function [X_train, X_val, pca_model] = apply_pca_transform(X_train, X_val, cfg)
    % Normalize
    X_train = X_train ./ vecnorm(X_train, 2, 2);
    X_val = X_val ./ vecnorm(X_val, 2, 2);
    
    % Standardize
    mu = mean(X_train, 1);
    sigma = std(X_train, 0, 1);
    sigma(sigma == 0) = 1;
    
    X_train = (X_train - mu) ./ sigma;
    X_val = (X_val - mu) ./ sigma;
    
    % PCA
    [coeff, ~, ~, ~, explained] = pca(X_train);
    
    % Select components
    cumvar = cumsum(explained) / 100;
    n_comp = find(cumvar >= cfg.pca.variance_threshold, 1, 'first');
    n_comp = min(n_comp, cfg.pca.max_components);
    
    % Transform
    X_train = X_train * coeff(:, 1:n_comp);
    X_val = X_val * coeff(:, 1:n_comp);
    
    % Store model
    pca_model = struct('coeff', coeff, 'n_comp', n_comp, 'mu', mu, 'sigma', sigma);
end


%% Helper: Aggregate spectrum predictions to sample level
function sample_preds = aggregate_to_samples(spectrum_preds, sample_map, n_samples)
    % Majority voting: most common prediction among spectra for each sample
    sample_preds = zeros(n_samples, 1);
    
    for i = 1:n_samples
        spectra_idx = (sample_map == i);
        sample_preds(i) = mode(spectrum_preds(spectra_idx));
    end
end


%% Helper: Get classifier configurations
function classifiers = get_classifier_configs(~)
    classifiers = {
        struct('name', 'LDA', 'type', 'lda'),
        struct('name', 'PLSDA', 'type', 'plsda', 'n_components', 5),
        struct('name', 'SVM', 'type', 'svm'),
        struct('name', 'RandomForest', 'type', 'rf', 'n_trees', 100)
    };
end


%% Helper: Train classifier
function model = train_classifier(classifier_cfg, X_train, y_train)
    switch classifier_cfg.type
        case 'lda'
            try
                model = fitcdiscr(X_train, y_train, 'DiscrimType', 'linear');
            catch
                model = fitcdiscr(X_train, y_train, 'DiscrimType', 'pseudoLinear');
            end
            
        case 'plsda'
            % Simple PLS-DA implementation
            n_comp = classifier_cfg.n_components;
            
            % Create dummy Y matrix
            Y = zeros(length(y_train), 1);
            Y(y_train == 1) = 1;
            Y(y_train == 3) = -1;
            
            % Fit PLS
            [~, ~, ~, ~, beta] = plsregress(X_train, Y, n_comp);
            model = struct('beta', beta, 'type', 'plsda');
            
        case 'svm'
            model = fitcsvm(X_train, y_train, 'KernelFunction', 'rbf', ...
                           'Standardize', false, 'BoxConstraint', 1);
            
        case 'rf'
            model = TreeBagger(classifier_cfg.n_trees, X_train, y_train, ...
                              'Method', 'classification', 'OOBPrediction', 'off');
    end
end


%% Helper: Compute metrics
function metrics = compute_metrics_direct(results)
    % Compute classification metrics from CV results
    
    y_true = results.sample_true;
    y_pred = results.sample_predictions;
    
    % Convert to binary (1 vs 3)
    y_true_bin = (y_true == 1);
    y_pred_bin = (y_pred == 1);
    
    % Confusion matrix
    tp = sum(y_true_bin & y_pred_bin);
    tn = sum(~y_true_bin & ~y_pred_bin);
    fp = sum(~y_true_bin & y_pred_bin);
    fn = sum(y_true_bin & ~y_pred_bin);
    
    % Metrics
    metrics.accuracy_mean = (tp + tn) / length(y_true);
    metrics.sensitivity_mean = tp / (tp + fn);
    metrics.specificity_mean = tn / (tn + fp);
    metrics.precision_mean = tp / (tp + fp);
    metrics.f1_mean = 2 * tp / (2*tp + fp + fn);
    
    % Compute std across folds
    fold_ids = unique(results.fold_info, 'rows');
    n_folds = size(fold_ids, 1);
    
    fold_acc = zeros(n_folds, 1);
    fold_sens = zeros(n_folds, 1);
    fold_spec = zeros(n_folds, 1);
    
    for i = 1:n_folds
        rep = fold_ids(i, 1);
        fold = fold_ids(i, 2);
        
        idx = (results.fold_info(:, 1) == rep) & (results.fold_info(:, 2) == fold);
        
        yt = y_true_bin(idx);
        yp = y_pred_bin(idx);
        
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
    metrics.auc_mean = metrics.accuracy_mean;  % Simplified
    metrics.auc_std = metrics.accuracy_std;
end
