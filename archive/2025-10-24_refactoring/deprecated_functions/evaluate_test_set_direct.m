%% EVALUATE_TEST_SET_DIRECT - Evaluate trained model on held-out test set
%
% Trains a final model on all training data and evaluates performance on 
% an independent test set. Implements cost-sensitive learning to prioritize
% WHO-3 tumor detection.
%
% SYNTAX:
%   test_results = evaluate_test_set_direct(data, cfg, best_classifier_name)
%
% INPUTS:
%   data                - Structure with 'train' and 'test' fields containing:
%                         * spectra: Cell array of spectral matrices
%                         * labels: WHO grades (1 or 3)
%                         * diss_id: Sample IDs
%                         * patient_id: Patient IDs
%                         * n_samples: Number of samples
%                         * total_spectra: Total number of spectra
%   cfg                 - Configuration structure from config.m
%   best_classifier_name- Classifier to use: 'LDA', 'PLSDA', 'SVM', 'RandomForest'
%                         (default: 'RandomForest')
%
% OUTPUTS:
%   test_results        - Structure containing:
%                         * spectrum_predictions: Predictions at spectrum level
%                         * spectrum_metrics: Performance metrics (spectrum level)
%                         * sample_predictions: Predictions at sample level (majority vote)
%                         * sample_metrics: Performance metrics (sample level)
%                         * final_model: Trained model object
%                         * pca_model: PCA model (if LDA) or empty
%                         * std_params: Standardization parameters
%
% NOTES:
%   - PCA is ONLY applied for LDA classifier
%   - All classifiers use z-score standardized spectra
%   - Cost-sensitive learning applied during training to prioritize WHO-3 detection
%   - Sample-level predictions use majority voting across spectra
%
% EXAMPLE:
%   cfg = config();
%   data = load_data_direct(cfg);
%   results = evaluate_test_set_direct(data, cfg, 'SVM');
%
% See also: load_data_direct, config, run_patientwise_cv_direct

function test_results = evaluate_test_set_direct(data, cfg, best_classifier_name)
    fprintf('\n=== EVALUATING TEST SET ===\n');
    
    % Default to RandomForest if not specified
    if nargin < 3
        best_classifier_name = 'RandomForest';
    end
    
    %% Train final model on ALL training data
    fprintf('\n1. Training final model on all training data...\n');
    
    train = data.train;
    test = data.test;
    
    fprintf('Training set: %d samples, %d spectra\n', train.n_samples, train.total_spectra);
    fprintf('Test set: %d samples, %d spectra\n', test.n_samples, test.total_spectra);
    
    % Extract all training spectra
    [X_train, y_train] = extract_all_spectra(train, 1:train.n_samples);
    fprintf('  Extracted %d training spectra\n', size(X_train, 1));
    
    % Standardize training data (for all classifiers)
    [X_train_std, std_params] = standardize_spectra_train(X_train);
    
    % Get classifier configuration
    classifier_cfg = get_classifier_config(best_classifier_name, cfg);
    
    % Apply PCA ONLY if using LDA
    if strcmp(classifier_cfg.type, 'lda')
        fprintf('  Applying PCA for LDA...\n');
        [X_train_feat, ~, pca_model] = apply_pca_transform_train(X_train_std, cfg);
        fprintf('  PCA: %d features -> %d components (%.2f%% variance)\n', ...
            size(X_train_std, 2), pca_model.n_comp, sum(pca_model.explained(1:pca_model.n_comp)));
        use_pca = true;
    else
        fprintf('  Using original standardized spectra (NO PCA for %s)\n', best_classifier_name);
        X_train_feat = X_train_std;
        pca_model = [];
        use_pca = false;
    end
    
    % Train best classifier
    fprintf('  Training %s...\n', best_classifier_name);
    final_model = train_classifier(classifier_cfg, X_train_feat, y_train);
    
    %% Extract test spectra
    fprintf('\n2. Extracting test data...\n');
    [X_test, y_test, test_sample_map] = extract_all_spectra_with_map(test, 1:test.n_samples);
    fprintf('  Extracted %d test spectra from %d samples\n', size(X_test, 1), test.n_samples);
    
    % Standardize test data using training parameters
    X_test_std = standardize_spectra_test(X_test, std_params);
    
    % Apply PCA transform only if LDA
    if use_pca
        X_test_feat = apply_pca_transform_test(X_test_std, pca_model);
    else
        X_test_feat = X_test_std;
    end
    
    %% Make predictions on test set
    fprintf('\n3. Making predictions...\n');
    
    % Handle different model types
    if isstruct(final_model) && isfield(final_model, 'type') && strcmp(final_model.type, 'plsda')
        % PLSDA prediction
        scores_raw = [ones(size(X_test_feat, 1), 1), X_test_feat] * final_model.beta;
        spectrum_preds = scores_raw > 0;
        spectrum_preds = double(spectrum_preds);
    else
        % Standard MATLAB model
        [spectrum_preds, ~] = predict(final_model, X_test_feat);
        
        % Handle RandomForest cell array output
        if iscell(spectrum_preds)
            spectrum_preds = cellfun(@str2double, spectrum_preds);
        end
    end
    
    fprintf('  Predicted %d spectra\n', length(spectrum_preds));
    
    %% Aggregate predictions per sample (MAJORITY VOTING)
    fprintf('  Aggregating to sample-level predictions via MAJORITY VOTE...\n');
    sample_preds = aggregate_to_samples(spectrum_preds, test_sample_map, test.n_samples);
    
    %% Compute metrics at both levels
    fprintf('\n4. Computing performance metrics...\n');
    
    % SPECTRUM-LEVEL metrics
    spectrum_metrics = compute_binary_metrics(y_test, spectrum_preds);
    
    fprintf('\n--- SPECTRUM-LEVEL RESULTS (n=%d) ---\n', length(y_test));
    fprintf('  Accuracy:    %.1f%%\n', spectrum_metrics.accuracy * 100);
    fprintf('  Sensitivity: %.1f%%\n', spectrum_metrics.sensitivity * 100);
    fprintf('  Specificity: %.1f%%\n', spectrum_metrics.specificity * 100);
    fprintf('  Precision:   %.1f%%\n', spectrum_metrics.precision * 100);
    fprintf('  F1-Score:    %.3f\n', spectrum_metrics.f1);
    
    % SAMPLE-LEVEL metrics
    sample_metrics = compute_binary_metrics(test.labels, sample_preds);
    
    fprintf('\n--- SAMPLE-LEVEL RESULTS (n=%d) ---\n', test.n_samples);
    fprintf('  (Aggregated via MAJORITY VOTE of spectrum predictions)\n');
    fprintf('  Accuracy:    %.1f%%\n', sample_metrics.accuracy * 100);
    fprintf('  Sensitivity: %.1f%%\n', sample_metrics.sensitivity * 100);
    fprintf('  Specificity: %.1f%%\n', sample_metrics.specificity * 100);
    fprintf('  Precision:   %.1f%%\n', sample_metrics.precision * 100);
    fprintf('  F1-Score:    %.3f\n', sample_metrics.f1);
    
    % Confusion matrix
    fprintf('\n--- CONFUSION MATRIX (Sample-Level) ---\n');
    print_confusion_matrix(test.labels, sample_preds);
    
    %% Store results
    test_results = struct();
    test_results.spectrum_predictions = spectrum_preds;
    test_results.spectrum_true = y_test;
    test_results.spectrum_metrics = spectrum_metrics;
    
    test_results.sample_predictions = sample_preds;
    test_results.sample_true = test.labels;
    test_results.sample_metrics = sample_metrics;
    test_results.sample_ids = test.diss_id;
    test_results.patient_ids = test.patient_id;
    test_results.aggregation_method = 'majority_vote';  % Document how samples were aggregated
    
    test_results.final_model = final_model;
    test_results.pca_model = pca_model;
    test_results.std_params = std_params;
    test_results.classifier_name = best_classifier_name;
    test_results.used_pca = use_pca;
    
    %% Save results
    fprintf('\n5. Saving results...\n');
    results_file = fullfile(cfg.paths.results, 'test_results_direct.mat');
    save(results_file, 'test_results', '-v7.3');
    fprintf('  Saved to: %s\n', results_file);
    
    fprintf('\n=== TEST SET EVALUATION COMPLETE ===\n');
end


%% ========================================================================
%  HELPER FUNCTIONS
% =========================================================================

function [X, y] = extract_all_spectra(data, sample_indices)
    % Extract all spectra from specified samples
    % Concatenates spectra from multiple samples into single matrices
    X = [];
    y = [];
    for i = 1:length(sample_indices)
        idx = sample_indices(i);
        spectra = data.spectra{idx};
        labels = repmat(data.labels(idx), size(spectra, 1), 1);
        X = [X; spectra];
        y = [y; labels];
    end
end


function [X, y, sample_map] = extract_all_spectra_with_map(data, sample_indices)
    % Extract all spectra with sample mapping for aggregation
    % Returns sample_map indicating which sample each spectrum belongs to
    X = [];
    y = [];
    sample_map = [];
    for i = 1:length(sample_indices)
        idx = sample_indices(i);
        spectra = data.spectra{idx};
        n_spec = size(spectra, 1);
        labels = repmat(data.labels(idx), n_spec, 1);
        X = [X; spectra];
        y = [y; labels];
        sample_map = [sample_map; repmat(i, n_spec, 1)];
    end
end


function [X_train_pca, X_val_pca, pca_model] = apply_pca_transform_train(X_train, cfg)
    % Apply PCA for dimensionality reduction (LDA ONLY)
    % Determines number of components based on variance threshold
    % Input should already be z-score standardized
    
    [coeff, ~, ~, ~, explained] = pca(X_train);
    
    % Determine number of components to retain
    cumvar = cumsum(explained);
    n_comp = find(cumvar >= cfg.pca.variance_threshold * 100, 1, 'first');
    n_comp = min(n_comp, cfg.pca.max_components);
    
    % Project training data
    X_train_pca = X_train * coeff(:, 1:n_comp);
    X_val_pca = [];  % Not used in this context
    
    % Store PCA model for test set transformation
    pca_model = struct('coeff', coeff, 'n_comp', n_comp, 'explained', explained);
end


function X_test_pca = apply_pca_transform_test(X_test, pca_model)
    % Apply PCA transformation using training-derived parameters
    % Input should already be z-score standardized
    X_test_pca = X_test * pca_model.coeff(:, 1:pca_model.n_comp);
end


function [X_std, params] = standardize_spectra_train(X)
    % Z-score standardization of training spectra
    % Returns standardized data and parameters for test set
    mu = mean(X, 1);
    sigma = std(X, 0, 1);
    sigma(sigma == 0) = 1;  % Avoid division by zero
    X_std = (X - mu) ./ sigma;
    params = struct('mu', mu, 'sigma', sigma);
end


function X_std = standardize_spectra_test(X, params)
    % Apply z-score standardization using training parameters
    X_std = (X - params.mu) ./ params.sigma;
end


function cfg_out = get_classifier_config(classifier_name, cfg)
    % Get classifier configuration including cost-sensitive parameters
    %
    % Cost penalty is applied during training to penalize misclassification
    % of WHO-3 tumors (malignant) more heavily than WHO-1 tumors
    
    % Get cost penalty from config (default: 5 = 5x penalty for missing WHO-3)
    cost_penalty = 5;
    if nargin >= 2 && isfield(cfg, 'classifiers') && isfield(cfg.classifiers, 'cost_who3_penalty')
        cost_penalty = cfg.classifiers.cost_who3_penalty;
    end
    
    switch classifier_name
        case 'LDA'
            cfg_out = struct('name', 'LDA', 'type', 'lda', 'cost_penalty', cost_penalty);
        case 'PLSDA'
            cfg_out = struct('name', 'PLSDA', 'type', 'plsda', 'n_components', 5, 'cost_penalty', cost_penalty);
        case 'SVM'
            cfg_out = struct('name', 'SVM', 'type', 'svm', 'cost_penalty', cost_penalty);
        case 'RandomForest'
            cfg_out = struct('name', 'RandomForest', 'type', 'rf', 'n_trees', 100, 'cost_penalty', cost_penalty);
        otherwise
            error('Unknown classifier: %s', classifier_name);
    end
end


function model = train_classifier(classifier_cfg, X_train, y_train)
    % Train classifier with cost-sensitive learning
    %
    % Cost-sensitive methods prioritize detection of WHO-3 tumors:
    %   - LDA: Weighted class priors
    %   - SVM: Asymmetric cost matrix
    %   - RandomForest: Sample weights
    %   - PLSDA: Cost penalty stored for reference
    
    % Extract cost penalty
    cost_penalty = 5;  % Default
    if isfield(classifier_cfg, 'cost_penalty')
        cost_penalty = classifier_cfg.cost_penalty;
    end
    
    switch classifier_cfg.type
        case 'lda'
            try
                % Cost-sensitive LDA via weighted class priors
                n_who1 = sum(y_train == 1);
                n_who3 = sum(y_train == 3);
                
                % Adjust priors to emphasize WHO-3 detection
                prior_who1 = n_who1 / (n_who1 + cost_penalty * n_who3);
                prior_who3 = (cost_penalty * n_who3) / (n_who1 + cost_penalty * n_who3);
                
                model = fitcdiscr(X_train, y_train, ...
                    'DiscrimType', 'linear', ...
                    'Prior', [prior_who1; prior_who3]);
            catch
                % Fallback if singular covariance matrix
                model = fitcdiscr(X_train, y_train, 'DiscrimType', 'pseudoLinear');
            end
            
        case 'plsda'
            % PLS Discriminant Analysis (regression-based approach)
            n_comp = classifier_cfg.n_components;
            Y = zeros(length(y_train), 1);
            Y(y_train == 1) = 1;   % WHO-1
            Y(y_train == 3) = -1;  % WHO-3
            [~, ~, ~, ~, beta] = plsregress(X_train, Y, n_comp);
            model = struct('beta', beta, 'type', 'plsda', 'cost_penalty', cost_penalty);
            
        case 'svm'
            % Cost-sensitive SVM via asymmetric cost matrix
            % Cost(i,j) = cost of predicting class j when true class is i
            % [0, 1; cost_penalty, 0] penalizes WHO-3->WHO-1 errors heavily
            cost_matrix = [0, 1; cost_penalty, 0];
            
            model = fitcsvm(X_train, y_train, ...
                           'KernelFunction', 'rbf', ...
                           'Standardize', false, ...  % Already standardized
                           'KernelScale', 'auto', ...
                           'BoxConstraint', 1, ...
                           'Cost', cost_matrix);
            
        case 'rf'
            % Cost-sensitive Random Forest via sample weights
            sample_weights = ones(length(y_train), 1);
            sample_weights(y_train == 3) = cost_penalty;  % WHO-3 samples weighted more
            
            model = TreeBagger(classifier_cfg.n_trees, X_train, y_train, ...
                              'Method', 'classification', ...
                              'OOBPrediction', 'off', ...
                              'Weights', sample_weights);
        otherwise
            error('Unknown classifier type: %s', classifier_cfg.type);
    end
end


function sample_preds = aggregate_to_samples(spectrum_preds, sample_map, n_samples)
    % Aggregate spectrum-level predictions to sample-level via majority voting
    % Each sample's prediction is the mode of its spectra predictions
    sample_preds = zeros(n_samples, 1);
    for s = 1:n_samples
        sample_spectra_preds = spectrum_preds(sample_map == s);
        sample_preds(s) = mode(sample_spectra_preds);
    end
end


function metrics = compute_binary_metrics(y_true, y_pred)
    % Compute binary classification metrics
    % Assumes WHO-1 is positive class (benign) and WHO-3 is negative class (malignant)
    y_true_bin = (y_true == 1);
    y_pred_bin = (y_pred == 1);
    
    tp = sum(y_true_bin & y_pred_bin);    % True WHO-1
    tn = sum(~y_true_bin & ~y_pred_bin);  % True WHO-3
    fp = sum(~y_true_bin & y_pred_bin);   % WHO-3 predicted as WHO-1
    fn = sum(y_true_bin & ~y_pred_bin);   % WHO-1 predicted as WHO-3
    
    metrics.accuracy = (tp + tn) / length(y_true);
    metrics.sensitivity = tp / max(tp + fn, 1);  % WHO-1 detection rate
    metrics.specificity = tn / max(tn + fp, 1);  % WHO-3 detection rate
    metrics.precision = tp / max(tp + fp, 1);
    metrics.f1 = 2 * tp / max(2*tp + fp + fn, 1);
    
    metrics.tp = tp;
    metrics.tn = tn;
    metrics.fp = fp;
    metrics.fn = fn;
end


function print_confusion_matrix(y_true, y_pred)
    % Print confusion matrix for binary classification (WHO-1 vs WHO-3)
    y_true_bin = (y_true == 1);
    y_pred_bin = (y_pred == 1);
    
    tp = sum(y_true_bin & y_pred_bin);
    tn = sum(~y_true_bin & ~y_pred_bin);
    fp = sum(~y_true_bin & y_pred_bin);
    fn = sum(y_true_bin & ~y_pred_bin);
    
    fprintf('                 Predicted Grade 1  Predicted Grade 3\n');
    fprintf('True Grade 1          %4d               %4d\n', tp, fn);
    fprintf('True Grade 3          %4d               %4d\n', fp, tn);
    fprintf('\n');
    fprintf('  TP=%d, TN=%d, FP=%d, FN=%d\n', tp, tn, fp, fn);
end
