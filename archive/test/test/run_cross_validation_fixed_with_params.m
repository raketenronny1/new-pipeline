%% PHASE 3: MODEL SELECTION VIA CROSS-VALIDATION
% This script performs model selection using cross-validation on the training set
% Fixed version that properly adds optimal_params to cv_results

function cv_results = run_cross_validation_fixed_with_params(cfg)
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

    %% Define Hyperparameter Grids
    % PLS-DA: Number of components
    plsda_n_comp_grid = 1:min(10, size(X_train_pca, 2));

    % SVM: C and gamma
    svm_C_grid = 10.^(-2:0.5:2);  % [0.01, 0.03, ..., 100]
    svm_gamma_grid = 10.^(-3:0.5:1);  % [0.001, 0.003, ..., 10]

    % Random Forest: n_trees and max_depth
    rf_n_trees_grid = [50, 100, 200, 500];
    rf_max_depth_grid = [5, 10, 20, 30];

    %% Cross-Validation Loop
    for clf_idx = 1:4
        fprintf('\nEvaluating %s...\n', classifierNames{clf_idx});
        
        cv_results{clf_idx} = struct();
        cv_results{clf_idx}.classifier = classifierNames{clf_idx};
        cv_results{clf_idx}.performance = cell(n_repeats, 1);
        
        % Initialize optimal parameters structure for each classifier
        switch classifierNames{clf_idx}
            case 'LDA'
                cv_results{clf_idx}.optimal_params = struct();
                % LDA doesn't have parameters to optimize
            case 'PLSDA'
                % Find optimal number of components
                best_comp = optimize_plsda(X_train_pca, trainingData.y, plsda_n_comp_grid);
                cv_results{clf_idx}.optimal_params = struct('n_components', best_comp);
            case 'SVM'
                % Find optimal C and gamma
                [best_C, best_gamma] = optimize_svm(X_train_pca, trainingData.y, svm_C_grid, svm_gamma_grid);
                cv_results{clf_idx}.optimal_params = struct('C', best_C, 'gamma', best_gamma);
            case 'RandomForest'
                % Find optimal number of trees and max depth
                [best_n_trees, best_max_depth] = optimize_rf(X_train_pca, trainingData.y, rf_n_trees_grid, rf_max_depth_grid);
                cv_results{clf_idx}.optimal_params = struct('n_trees', best_n_trees, 'max_depth', best_max_depth);
        end
        
        for rep = 1:n_repeats
            if mod(rep, 10) == 0
                fprintf('Repeat %d/%d\n', rep, n_repeats);
            end
            
            % Create stratified partition
            cv_partition = cvpartition(trainingData.y, 'KFold', n_folds);
            
            fold_results = struct();
            fold_results.predictions = cell(n_folds, 1);
            fold_results.scores = cell(n_folds, 1);
            fold_results.true_labels = cell(n_folds, 1);
            
            for fold = 1:n_folds
                % Get training and validation sets for this fold
                train_idx = training(cv_partition, fold);
                val_idx = test(cv_partition, fold);
                
                X_train_fold = X_train_pca(train_idx, :);
                y_train_fold = trainingData.y(train_idx);
                X_val_fold = X_train_pca(val_idx, :);
                y_val_fold = trainingData.y(val_idx);
                
                % Train and evaluate classifier
                switch classifierNames{clf_idx}
                    case 'LDA'
                        % Linear Discriminant Analysis
                        try
                            % First try linear discriminant
                            model = fitcdiscr(X_train_fold, y_train_fold, 'DiscrimType', 'linear');
                        catch ME
                            if contains(ME.identifier, 'ZeroDiagCovLin')
                                % If there's a zero variance predictor, use pseudolinear instead
                                warning('Zero variance predictor detected. Using pseudolinear discriminant instead.');
                                model = fitcdiscr(X_train_fold, y_train_fold, 'DiscrimType', 'pseudoLinear');
                            else
                                % If it's a different error, rethrow it
                                rethrow(ME);
                            end
                        end
                        [predictions, scores] = predict(model, X_val_fold);
                        
                    case 'PLSDA'
                        % Use optimal number of components
                        n_comp = cv_results{clf_idx}.optimal_params.n_components;
                        
                        % Convert labels to numeric if they're not already
                        if iscategorical(y_train_fold)
                            % Create a dummy variable representation for PLS
                            % (1 for WHO-1, -1 for WHO-3)
                            numLabels = zeros(length(y_train_fold), 1);
                            numLabels(y_train_fold == 'WHO-1' | double(y_train_fold) == 1) = 1;
                            numLabels(y_train_fold == 'WHO-3' | double(y_train_fold) == 3) = -1;
                        else
                            % Assume numeric (1 for WHO-1, 3 for WHO-3)
                            numLabels = zeros(length(y_train_fold), 1);
                            numLabels(y_train_fold == 1) = 1;
                            numLabels(y_train_fold == 3) = -1;
                        end
                        
                        % Call the custom PLS function
                        [T, P, U, Q, B, W] = pls(X_train_fold, numLabels, 1e-6);
                        
                        % Keep only the optimal number of components
                        T = T(:, 1:min(n_comp, size(T, 2)));
                        P = P(:, 1:min(n_comp, size(P, 2)));
                        B = B(1:min(n_comp, size(B, 1)), 1:min(n_comp, size(B, 2)));
                        Q = Q(:, 1:min(n_comp, size(Q, 2)));
                        
                        % Project validation data onto PLS space
                        scores = X_val_fold * (P*B*Q');
                        
                        % Convert scores to predictions and probability scores
                        predictions = zeros(size(scores));
                        predictions(scores >= 0) = 1;  % WHO-1
                        predictions(scores < 0) = 3;   % WHO-3
                        
                        % Create pseudo-probability scores for ROC analysis
                        prob_scores = zeros(size(scores, 1), 2);
                        prob_scores(:, 1) = 1 ./ (1 + exp(-scores));      % WHO-1 probability
                        prob_scores(:, 2) = 1 ./ (1 + exp(scores));       % WHO-3 probability
                        
                        % Normalize
                        row_sums = sum(prob_scores, 2);
                        prob_scores = prob_scores ./ row_sums;
                        
                        scores = prob_scores;
                        
                    case 'SVM'
                        % Support Vector Machine
                        C = cv_results{clf_idx}.optimal_params.C;
                        gamma = cv_results{clf_idx}.optimal_params.gamma;
                        
                        model = fitcsvm(X_train_fold, y_train_fold, ...
                                     'KernelFunction', 'rbf', ...
                                     'BoxConstraint', C, ...
                                     'KernelScale', 1/sqrt(gamma));
                        
                        % Use score transformation for probability scores
                        score_model = fitPosterior(model);
                        [predictions, scores] = predict(score_model, X_val_fold);
                        
                    case 'RandomForest'
                        % Random Forest
                        n_trees = cv_results{clf_idx}.optimal_params.n_trees;
                        max_depth = cv_results{clf_idx}.optimal_params.max_depth;
                        
                        model = TreeBagger(n_trees, X_train_fold, y_train_fold, ...
                                        'Method', 'classification', ...
                                        'MaxNumSplits', max_depth);
                        
                        [pred_labels, scores] = predict(model, X_val_fold);
                        predictions = cellfun(@str2num, pred_labels);  % Convert string predictions to numbers
                end
                
                % Store results for this fold
                fold_results.predictions{fold} = predictions;
                fold_results.scores{fold} = scores;
                fold_results.true_labels{fold} = y_val_fold;
            end
            
            % Calculate performance metrics for this repeat
            cv_results{clf_idx}.performance{rep} = calculate_metrics(fold_results);
        end
    end

    %% Aggregate Results
    fprintf('\nAggregating results...\n');

    % Create summary table
    summary_table = table();
    summary_table.Classifier = categorical(classifierNames)';

    % Calculate mean and SD for each metric
    for clf_idx = 1:4
        perf_array = vertcat(cv_results{clf_idx}.performance{:});
        
        summary_table.Mean_Accuracy(clf_idx) = mean([perf_array.accuracy]);
        summary_table.SD_Accuracy(clf_idx) = std([perf_array.accuracy]);
        summary_table.Mean_Sensitivity_WHO3(clf_idx) = mean([perf_array.sensitivity]);
        summary_table.SD_Sensitivity_WHO3(clf_idx) = std([perf_array.sensitivity]);
        summary_table.Mean_Specificity_WHO1(clf_idx) = mean([perf_array.specificity]);
        summary_table.SD_Specificity_WHO1(clf_idx) = std([perf_array.specificity]);
        summary_table.Mean_F2_WHO3(clf_idx) = mean([perf_array.f2]);
        summary_table.SD_F2_WHO3(clf_idx) = std([perf_array.f2]);
        summary_table.Mean_AUC(clf_idx) = mean([perf_array.auc]);
        summary_table.SD_AUC(clf_idx) = std([perf_array.auc]);
    end

    %% Select Best Classifier
    [~, best_idx] = max(summary_table.Mean_F2_WHO3);
    best_classifier = summary_table.Classifier(best_idx);

    fprintf('\nBest classifier: %s\n', char(best_classifier));
    fprintf('Mean F2-score: %.3f ± %.3f\n', ...
            summary_table.Mean_F2_WHO3(best_idx), ...
            summary_table.SD_F2_WHO3(best_idx));

    %% Save Results
    best_model_info = struct();
    best_model_info.classifier = char(best_classifier);
    best_model_info.cv_performance = summary_table(best_idx, :);
    best_model_info.optimal_params = cv_results{best_idx}.optimal_params;

    save(fullfile(cfg.paths.results, 'cv_performance.mat'), 'summary_table');
    save(fullfile(cfg.paths.results, 'best_classifier_selection.mat'), 'best_model_info');

    fprintf('Model selection complete.\n');
end

%% Helper Functions

function metrics = calculate_metrics(fold_results)
    % Pre-allocate arrays for better performance
    total_samples = sum(cellfun(@length, fold_results.predictions));
    all_pred = zeros(total_samples, 1);
    all_true = zeros(total_samples, 1);
    all_scores = zeros(total_samples, size(fold_results.scores{1}, 2));
    
    % Combine all folds with pre-allocated arrays
    idx = 1;
    for i = 1:length(fold_results.predictions)
        n = length(fold_results.predictions{i});
        all_pred(idx:idx+n-1) = fold_results.predictions{i};
        all_true(idx:idx+n-1) = fold_results.true_labels{i};
        all_scores(idx:idx+n-1,:) = fold_results.scores{i};
        idx = idx + n;
    end
    
    % Cache confusion matrix
    cm = confusionmat(all_true, all_pred);
    
    % Calculate metrics using cached confusion matrix
    metrics = struct();
    metrics.accuracy = sum(diag(cm)) / sum(cm(:));
    metrics.sensitivity = cm(2,2) / sum(cm(2,:));  % WHO-3 recall
    metrics.specificity = cm(1,1) / sum(cm(1,:));  % WHO-1 specificity
    
    % F-scores
    precision = cm(2,2) / sum(cm(:,2));
    recall = metrics.sensitivity;
    metrics.f1 = 2 * (precision * recall) / (precision + recall);
    
    % F2-score (beta = 2)
    beta = 2;
    metrics.f2 = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall);
    
    % AUC-ROC
    % First determine what type of label we have (categorical, string, or numeric)
    if iscategorical(all_true)
        pos_class = categorical({'WHO-3'});
    elseif isstring(all_true) || iscell(all_true)
        pos_class = 'WHO-3';
    else
        % Numeric - assume 3 for WHO-3
        pos_class = 3;
    end
    
    % Debug info
    fprintf('AUC calculation: Using positive class of type %s\n', class(pos_class));
    
    try
        [~, ~, ~, metrics.auc] = perfcurve(all_true, all_scores(:,2), pos_class);
    catch ME
        warning('AUC:CalculationError', 'Could not calculate AUC: %s. Setting AUC to NaN.', ME.message);
        metrics.auc = NaN;
    end
end

function best_comp = optimize_plsda(X_train, y_train, n_comp_grid)
    % Use 3-fold CV to find optimal number of components
    cv_partition = cvpartition(y_train, 'KFold', 3);
    n_components = length(n_comp_grid);
    cv_scores = zeros(n_components, 1);
    
    for i = 1:n_components
        n_comp = n_comp_grid(i);
        fold_scores = zeros(3, 1);
        
        for k = 1:3
            % Get training and validation sets for this fold
            train_idx = training(cv_partition, k);
            val_idx = test(cv_partition, k);
            
            X_train_fold = X_train(train_idx, :);
            y_train_fold = y_train(train_idx);
            X_val_fold = X_train(val_idx, :);
            y_val_fold = y_train(val_idx);
            
            % Convert labels to numeric for PLS
            if iscategorical(y_train_fold)
                numLabels = zeros(length(y_train_fold), 1);
                numLabels(y_train_fold == 'WHO-1' | double(y_train_fold) == 1) = 1;
                numLabels(y_train_fold == 'WHO-3' | double(y_train_fold) == 3) = -1;
            else
                % Assume numeric
                numLabels = zeros(length(y_train_fold), 1);
                numLabels(y_train_fold == 1) = 1;
                numLabels(y_train_fold == 3) = -1;
            end
            
            % Call custom PLS function
            [~, P, ~, Q, B, ~] = pls(X_train_fold, numLabels, 1e-6);
            
            % Keep only specified number of components
            n_comp_actual = min(n_comp, size(P, 2));
            P = P(:, 1:n_comp_actual);
            B = B(1:n_comp_actual, 1:n_comp_actual);
            Q = Q(:, 1:n_comp_actual);
            
            % Project validation data
            scores = X_val_fold * (P*B*Q');
            
            % Convert scores to predictions
            yhat = zeros(size(scores));
            yhat(scores >= 0) = 1;  % WHO-1
            yhat(scores < 0) = 3;   % WHO-3
            
            % Calculate F2-score
            cm = confusionmat(y_val_fold, yhat);
            precision = cm(2,2) / sum(cm(:,2));
            recall = cm(2,2) / sum(cm(2,:));
            beta = 2;
            f2 = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall);
            
            fold_scores(k) = f2;
        end
        
        cv_scores(i) = mean(fold_scores);
    end
    
    [~, best_idx] = max(cv_scores);
    best_comp = n_comp_grid(best_idx);
end

function [best_C, best_gamma] = optimize_svm(X_train, y_train, C_grid, gamma_grid)
    % Grid search with 3-fold CV
    cv_partition = cvpartition(y_train, 'KFold', 3);
    [n_C, n_gamma] = deal(length(C_grid), length(gamma_grid));
    cv_scores = zeros(n_C, n_gamma);
    
    for i = 1:n_C
        C = C_grid(i);
        for j = 1:n_gamma
            gamma = gamma_grid(j);
            fold_scores = zeros(3, 1);
            
            for k = 1:3
                % Get fold data
                train_idx = training(cv_partition, k);
                val_idx = test(cv_partition, k);
                
                X_train_fold = X_train(train_idx, :);
                y_train_fold = y_train(train_idx);
                X_val_fold = X_train(val_idx, :);
                y_val_fold = y_train(val_idx);
                
                % Train with current parameters
                model = fitcsvm(X_train_fold, y_train_fold, ...
                              'KernelFunction', 'rbf', ...
                              'BoxConstraint', C, ...
                              'KernelScale', 1/sqrt(gamma));
                [yhat, ~] = predict(model, X_val_fold);
                
                % Calculate F2-score
                cm = confusionmat(y_val_fold, yhat);
                precision = cm(2,2) / sum(cm(:,2));
                recall = cm(2,2) / sum(cm(2,:));
                beta = 2;
                f2 = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall);
                
                fold_scores(k) = f2;
            end
            
            cv_scores(i, j) = mean(fold_scores);
        end
    end
    
    [~, best_idx] = max(cv_scores(:));
    [i_best, j_best] = ind2sub(size(cv_scores), best_idx);
    best_C = C_grid(i_best);
    best_gamma = gamma_grid(j_best);
end

function [best_n_trees, best_max_depth] = optimize_rf(X_train, y_train, n_trees_grid, max_depth_grid)
    % Grid search with 3-fold CV
    cv_partition = cvpartition(y_train, 'KFold', 3);
    [n_trees_len, n_depth_len] = deal(length(n_trees_grid), length(max_depth_grid));
    cv_scores = zeros(n_trees_len, n_depth_len);
    
    for i = 1:n_trees_len
        n_trees = n_trees_grid(i);
        for j = 1:n_depth_len
            max_depth = max_depth_grid(j);
            fold_scores = zeros(3, 1);
            
            for k = 1:3
                % Get fold data
                train_idx = training(cv_partition, k);
                val_idx = test(cv_partition, k);
                
                X_train_fold = X_train(train_idx, :);
                y_train_fold = y_train(train_idx);
                X_val_fold = X_train(val_idx, :);
                y_val_fold = y_train(val_idx);
                
                % Train with current parameters
                model = TreeBagger(n_trees, X_train_fold, y_train_fold, ...
                                 'Method', 'classification', ...
                                 'MaxNumSplits', max_depth);
                [yhat, ~] = predict(model, X_val_fold);
                yhat = cellfun(@str2num, yhat);  % Convert string predictions to numbers
                
                % Calculate F2-score
                cm = confusionmat(y_val_fold, yhat);
                precision = cm(2,2) / sum(cm(:,2));
                recall = cm(2,2) / sum(cm(2,:));
                beta = 2;
                f2 = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall);
                
                fold_scores(k) = f2;
            end
            
            cv_scores(i, j) = mean(fold_scores);
        end
    end
    
    [~, best_idx] = max(cv_scores(:));
    [i_best, j_best] = ind2sub(size(cv_scores), best_idx);
    best_n_trees = n_trees_grid(i_best);
    best_max_depth = max_depth_grid(j_best);
end