
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
n_folds = 5;
n_repeats = 50;

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
                    % Use custom PLS function
                    try
                        % Find optimal number of components
                        best_comp = optimize_plsda(X_train_fold, y_train_fold, plsda_n_comp_grid);
                        
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
                        T = T(:, 1:min(best_comp, size(T, 2)));
                        P = P(:, 1:min(best_comp, size(P, 2)));
                        B = B(1:min(best_comp, size(B, 1)), 1:min(best_comp, size(B, 2)));
                        Q = Q(:, 1:min(best_comp, size(Q, 2)));
                        
                        % Project validation data onto PLS space
                        scores = X_val_fold * (P*B*Q');
                        
                        % Convert scores to predictions and probability scores
                        predictions = zeros(size(scores));
                        predictions(scores >= 0) = 1;  % WHO-1
                        predictions(scores < 0) = 3;   % WHO-3
                        
                        % Create a scores matrix compatible with other classifiers
                        % Column 1: WHO-1 probability, Column 2: WHO-3 probability
                        probScores = zeros(length(scores), 2);
                        
                        % Convert raw scores to probabilities using sigmoid function
                        sigmoid = @(x) 1 ./ (1 + exp(-x));
                        probWHO1 = sigmoid(scores);
                        probScores(:, 1) = probWHO1;
                        probScores(:, 2) = 1 - probWHO1;
                        
                        scores = probScores;
                    catch ME
                        % Use LDA as fallback if PLS fails
                        warning('Error in PLS-DA: %s. Using LDA as fallback.', ME.message);
                        try
                            model = fitcdiscr(X_train_fold, y_train_fold, 'DiscrimType', 'linear');
                        catch ME2
                            if contains(ME2.identifier, 'ZeroDiagCovLin')
                                model = fitcdiscr(X_train_fold, y_train_fold, 'DiscrimType', 'pseudoLinear');
                            else
                                rethrow(ME);
                            end
                        end
                        [predictions, scores] = predict(model, X_val_fold);
                    end
                    
                case 'SVM'
                    % Grid search for C and gamma
                    [best_C, best_gamma] = optimize_svm(X_train_fold, y_train_fold, svm_C_grid, svm_gamma_grid);
                    model = fitcsvm(X_train_fold, y_train_fold, ...
                                  'KernelFunction', 'rbf', ...
                                  'BoxConstraint', best_C, ...
                                  'KernelScale', 1/sqrt(best_gamma));
                    [predictions, scores] = predict(model, X_val_fold);
                    
                case 'RandomForest'
                    % Grid search for n_trees and max_depth
                    [best_n_trees, best_max_depth] = optimize_rf(X_train_fold, y_train_fold, ...
                                                               rf_n_trees_grid, rf_max_depth_grid);
                    model = TreeBagger(best_n_trees, X_train_fold, y_train_fold, ...
                                    'Method', 'classification', ...
                                    'MaxNumSplits', best_max_depth);
                    [predictions, scores] = predict(model, X_val_fold);
                    predictions = categorical(predictions);
            end
            
            % Store results for this fold
            fold_results.predictions{fold} = predictions;
            fold_results.scores{fold} = scores;
            fold_results.true_labels{fold} = y_val_fold;
        end
        
        % Calculate metrics for this repeat
        cv_results{clf_idx}.performance{rep} = calculate_metrics(fold_results);
    end
end

%% Aggregate Results
fprintf('\nAggregating results...\n');

% Initialize summary table
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
    
    % Calculate 95% confidence intervals for F2
    summary_table.CI95_F2_lower(clf_idx) = prctile([perf_array.f2], 2.5);
    summary_table.CI95_F2_upper(clf_idx) = prctile([perf_array.f2], 97.5);
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

save('results/meningioma_ftir_pipeline/cv_performance.csv', 'summary_table');
save('results/meningioma_ftir_pipeline/best_classifier_selection.mat', 'best_model_info');

%% Create Performance Visualization
create_cv_performance_plots(cv_results, classifierNames);

fprintf('Model selection complete.\n');
    catch ME
        error('Error in run_cross_validation: %s\nLine %d: %s', ...
            ME.identifier, ME.stack(1).line, ME.message);
    end
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
        warning('Could not calculate AUC: %s. Setting AUC to NaN.', ME.message);
        metrics.auc = NaN;
    end
end

function best_comp = optimize_plsda(X_train, y_train, n_comp_grid)
    % Use the custom PLS function to find optimal number of components
    
    % Simple 3-fold CV for component selection with optimized performance
    cv_partition = cvpartition(y_train, 'KFold', 3);
    n_components = length(n_comp_grid);
    cv_scores = zeros(n_components, 1);
    beta = 2;  % F2-score parameter
    
    % Pre-compute fold indices
    train_indices = cell(3, 1);
    test_indices = cell(3, 1);
    for k = 1:3
        train_indices{k} = training(cv_partition, k);
        test_indices{k} = test(cv_partition, k);
    end
    
    % Vectorized operations for each number of components
    for i = 1:n_components
        n_comp = n_comp_grid(i);
        fold_scores = zeros(3, 1);
        
        for k = 1:3  % Sequential processing
            % Get pre-computed fold data
            X_train_fold = X_train(train_indices{k}, :);
            y_train_fold = y_train(train_indices{k});
            X_val_fold = X_train(test_indices{k}, :);
            y_val_fold = y_train(test_indices{k});
            
            % Train and evaluate with custom PLS
            try
                % Convert labels to numeric for PLS
                if iscategorical(y_train_fold)
                    numLabels = zeros(length(y_train_fold), 1);
                    numLabels(y_train_fold == 'WHO-1' | double(y_train_fold) == 1) = 1;
                    numLabels(y_train_fold == 'WHO-3' | double(y_train_fold) == 3) = -1;
                else
                    % Assume numeric (1 for WHO-1, 3 for WHO-3)
                    numLabels = zeros(length(y_train_fold), 1);
                    numLabels(y_train_fold == 1) = 1;
                    numLabels(y_train_fold == 3) = -1;
                end
                
                % Call custom PLS function
                [~, P, ~, Q, B, ~] = pls(X_train_fold, numLabels, 1e-6);
                
                % Keep only the specified number of components
                n_comp_actual = min(n_comp, size(P, 2));
                P = P(:, 1:n_comp_actual);
                B = B(1:n_comp_actual, 1:n_comp_actual);
                Q = Q(:, 1:n_comp_actual);
                
                % Project validation data onto PLS space
                scores = X_val_fold * (P*B*Q');
                
                % Convert scores to predictions
                yhat = zeros(size(scores));
                yhat(scores >= 0) = 1;  % WHO-1
                yhat(scores < 0) = 3;   % WHO-3
            catch ME
                warning('Error training PLS model: %s. Using LDA instead.', ME.message);
                try
                    model = fitcdiscr(X_train_fold, y_train_fold, 'DiscrimType', 'pseudoLinear');
                    [yhat, ~] = predict(model, X_val_fold);
                catch
                    % If LDA fails too, just make a random guess based on class proportions
                    rng(42 + k);  % Reproducible randomness
                    yhat = randsample([1, 3], length(y_val_fold), true, [0.5, 0.5]);
                end
            end
            
            % Vectorized metric calculation
            cm = confusionmat(y_val_fold, yhat);
            precision = cm(2,2) / sum(cm(:,2));
            recall = cm(2,2) / sum(cm(2,:));
            fold_scores(k) = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall);
        end
        cv_scores(i) = mean(fold_scores);
    end
    
    [~, best_idx] = max(cv_scores);
    best_comp = n_comp_grid(best_idx);
end

function [best_C, best_gamma] = optimize_svm(X_train, y_train, C_grid, gamma_grid)
    % Grid search with 3-fold CV and parallel processing
    cv_partition = cvpartition(y_train, 'KFold', 3);
    [n_C, n_gamma] = deal(length(C_grid), length(gamma_grid));
    cv_scores = zeros(n_C, n_gamma);
    beta = 2;  % F2-score parameter
    
    % Pre-compute fold indices
    train_indices = cell(3, 1);
    test_indices = cell(3, 1);
    for k = 1:3
        train_indices{k} = training(cv_partition, k);
        test_indices{k} = test(cv_partition, k);
    end
    
    % Prepare parameter combinations for parallel processing
    [C_mesh, gamma_mesh] = meshgrid(1:n_C, 1:n_gamma);
    param_pairs = [C_mesh(:), gamma_mesh(:)];
    n_pairs = size(param_pairs, 1);
    all_scores = zeros(n_pairs, 1);
    
    % Sequential processing of parameter combinations
    for pair_idx = 1:n_pairs
        i = param_pairs(pair_idx, 1);
        j = param_pairs(pair_idx, 2);
        C = C_grid(i);
        gamma = gamma_grid(j);
        fold_scores = zeros(3, 1);
        
        for k = 1:3
            % Get pre-computed fold data
            X_train_fold = X_train(train_indices{k}, :);
            y_train_fold = y_train(train_indices{k});
            X_val_fold = X_train(test_indices{k}, :);
            y_val_fold = y_train(test_indices{k});
            
            % Train with current parameters
            model = fitcsvm(X_train_fold, y_train_fold, ...
                          'KernelFunction', 'rbf', ...
                          'BoxConstraint', C, ...
                          'KernelScale', 1/sqrt(gamma));
            [yhat, ~] = predict(model, X_val_fold);
            
            % Vectorized metric calculation
            cm = confusionmat(y_val_fold, yhat);
            precision = cm(2,2) / sum(cm(:,2));
            recall = cm(2,2) / sum(cm(2,:));
            fold_scores(k) = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall);
        end
        all_scores(pair_idx) = mean(fold_scores);
    end
    
    % Reshape results back to grid form
    cv_scores = reshape(all_scores, n_gamma, n_C)';
    
    [~, best_idx] = max(cv_scores(:));
    [i_best, j_best] = ind2sub(size(cv_scores), best_idx);
    best_C = C_grid(i_best);
    best_gamma = gamma_grid(j_best);
end

function [best_n_trees, best_max_depth] = optimize_rf(X_train, y_train, n_trees_grid, max_depth_grid)
    % Grid search with 3-fold CV and parallel processing
    cv_partition = cvpartition(y_train, 'KFold', 3);
    [n_trees_len, n_depth_len] = deal(length(n_trees_grid), length(max_depth_grid));
    cv_scores = zeros(n_trees_len, n_depth_len);
    beta = 2;  % F2-score parameter
    
    % Pre-compute fold indices
    train_indices = cell(3, 1);
    test_indices = cell(3, 1);
    for k = 1:3
        train_indices{k} = training(cv_partition, k);
        test_indices{k} = test(cv_partition, k);
    end
    
    % Prepare parameter combinations for parallel processing
    [trees_mesh, depth_mesh] = meshgrid(1:n_trees_len, 1:n_depth_len);
    param_pairs = [trees_mesh(:), depth_mesh(:)];
    n_pairs = size(param_pairs, 1);
    all_scores = zeros(n_pairs, 1);
    
    % Sequential processing of parameter combinations
    for pair_idx = 1:n_pairs
        i = param_pairs(pair_idx, 1);
        j = param_pairs(pair_idx, 2);
        n_trees = n_trees_grid(i);
        max_depth = max_depth_grid(j);
        fold_scores = zeros(3, 1);
        
        % Cache for confusion matrices
        cms = zeros(2, 2, 3);  % 2x2 confusion matrix for each fold
        
        for k = 1:3
            % Get pre-computed fold data
            X_train_fold = X_train(train_indices{k}, :);
            y_train_fold = y_train(train_indices{k});
            X_val_fold = X_train(test_indices{k}, :);
            y_val_fold = y_train(test_indices{k});
            
            % Train with optimized parameters
            model = TreeBagger(n_trees, X_train_fold, y_train_fold, ...
                             'Method', 'classification', ...
                             'MaxNumSplits', max_depth, ...
                             'NumPredictorsToSample', 'all', ...  % Optimize feature sampling
                             'OOBPrediction', 'off');            % Disable OOB to speed up training
            [yhat, ~] = predict(model, X_val_fold);
            yhat = categorical(yhat);
            
            % Cache confusion matrix
            cms(:,:,k) = confusionmat(y_val_fold, yhat);
            
            % Vectorized metric calculation using cached confusion matrix
            cm = cms(:,:,k);
            precision = cm(2,2) / sum(cm(:,2));
            recall = cm(2,2) / sum(cm(2,:));
            fold_scores(k) = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall);
        end
        all_scores(pair_idx) = mean(fold_scores);
    end
    
    % Reshape results back to grid form
    cv_scores = reshape(all_scores, n_depth_len, n_trees_len)';
    
    [~, best_idx] = max(cv_scores(:));
    [i_best, j_best] = ind2sub(size(cv_scores), best_idx);
    best_n_trees = n_trees_grid(i_best);
    best_max_depth = max_depth_grid(j_best);
end

function create_cv_performance_plots(cv_results, classifierNames)
    % Create boxplots of key metrics
    metrics = {'F2-Score (WHO-3)', 'Sensitivity (WHO-3)', 'Specificity (WHO-1)', 'AUC-ROC'};
    figure('Position', [100, 100, 1200, 800]);
    
    for i = 1:4
        subplot(2,2,i);
        
        % Collect metric values for each classifier
        data = cell(4,1);
        for clf = 1:4
            perf_array = vertcat(cv_results{clf}.performance{:});
            switch metrics{i}
                case 'F2-Score (WHO-3)'
                    data{clf} = [perf_array.f2];
                case 'Sensitivity (WHO-3)'
                    data{clf} = [perf_array.sensitivity];
                case 'Specificity (WHO-1)'
                    data{clf} = [perf_array.specificity];
                case 'AUC-ROC'
                    data{clf} = [perf_array.auc];
            end
        end
        
        % Create boxplot
        boxplot(cell2mat(data), repmat(categorical(classifierNames), [50 1]));
        ylabel(metrics{i});
        title(metrics{i});
        grid on;
    end
    
    % Save plot
    saveas(gcf, fullfile(cfg.paths.results, 'cv_boxplots.png'));
end