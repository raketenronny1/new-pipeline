
%% PHASE 3: MODEL SELECTION VIA CROSS-VALIDATION
% This script performs model selection using cross-validation on the training set

function run_cross_validation(cfg)
    clear; clc;
    addpath('src/meningioma_ftir_pipeline');

    %% Load Data
    fprintf('Loading transformed training data...\n');
    load(fullfile(cfg.paths.results, 'X_train_pca.mat'), 'X_train_pca');
    load(fullfile(cfg.paths.results, 'preprocessed_data.mat'), 'trainingData');

%% Set Up Cross-Validation
fprintf('Setting up cross-validation...\n');

% Set random seed for reproducibility
rng(42);

% CV parameters
n_folds = 5;
n_repeats = 50;

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
                    model = fitcdiscr(X_train_fold, y_train_fold, 'DiscrimType', 'linear');
                    [predictions, scores] = predict(model, X_val_fold);
                    
                case 'PLSDA'
                    % Find optimal number of components
                    best_comp = optimize_plsda(X_train_fold, y_train_fold, plsda_n_comp_grid);
                    model = fitcpls(X_train_fold, y_train_fold, 'NumComponents', best_comp);
                    [predictions, scores] = predict(model, X_val_fold);
                    
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

%% Helper Functions

function metrics = calculate_metrics(fold_results)
    % Combine all folds
    all_pred = vertcat(fold_results.predictions{:});
    all_true = vertcat(fold_results.true_labels{:});
    all_scores = vertcat(fold_results.scores{:});
    
    % Confusion matrix
    cm = confusionmat(all_true, all_pred);
    
    % Calculate metrics
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
    [~, ~, ~, metrics.auc] = perfcurve(all_true, all_scores(:,2), 'WHO-3');
end

function best_comp = optimize_plsda(X_train, y_train, n_comp_grid)
    % Simple 3-fold CV for component selection
    cv_partition = cvpartition(y_train, 'KFold', 3);
    cv_scores = zeros(length(n_comp_grid), 1);
    
    for i = 1:length(n_comp_grid)
        n_comp = n_comp_grid(i);
        fold_scores = zeros(3, 1);
        
        for fold = 1:3
            % Get fold data
            X_train_fold = X_train(training(cv_partition, fold), :);
            y_train_fold = y_train(training(cv_partition, fold));
            X_val_fold = X_train(test(cv_partition, fold), :);
            y_val_fold = y_train(test(cv_partition, fold));
            
            % Train and evaluate
            model = fitcpls(X_train_fold, y_train_fold, 'NumComponents', n_comp);
            [yhat, ~] = predict(model, X_val_fold);
            
            % Calculate F2 score
            cm = confusionmat(y_val_fold, yhat);
            precision = cm(2,2) / sum(cm(:,2));
            recall = cm(2,2) / sum(cm(2,:));
            beta = 2;
            f2 = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall);
            
            fold_scores(fold) = f2;
        end
        cv_scores(i) = mean(fold_scores);
    end
    
    [~, best_idx] = max(cv_scores);
    best_comp = n_comp_grid(best_idx);
end

function [best_C, best_gamma] = optimize_svm(X_train, y_train, C_grid, gamma_grid)
    % Grid search with 3-fold CV
    cv_partition = cvpartition(y_train, 'KFold', 3);
    cv_scores = zeros(length(C_grid), length(gamma_grid));
    
    for i = 1:length(C_grid)
        for j = 1:length(gamma_grid)
            fold_scores = zeros(3, 1);
            
            for fold = 1:3
                % Get fold data
                X_train_fold = X_train(training(cv_partition, fold), :);
                y_train_fold = y_train(training(cv_partition, fold));
                X_val_fold = X_train(test(cv_partition, fold), :);
                y_val_fold = y_train(test(cv_partition, fold));
                
                % Train and evaluate
                model = fitcsvm(X_train_fold, y_train_fold, ...
                              'KernelFunction', 'rbf', ...
                              'BoxConstraint', C_grid(i), ...
                              'KernelScale', 1/sqrt(gamma_grid(j)));
                [yhat, ~] = predict(model, X_val_fold);
                
                % Calculate F2 score
                cm = confusionmat(y_val_fold, yhat);
                precision = cm(2,2) / sum(cm(:,2));
                recall = cm(2,2) / sum(cm(2,:));
                beta = 2;
                f2 = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall);
                
                fold_scores(fold) = f2;
            end
            cv_scores(i,j) = mean(fold_scores);
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
    cv_scores = zeros(length(n_trees_grid), length(max_depth_grid));
    
    for i = 1:length(n_trees_grid)
        for j = 1:length(max_depth_grid)
            fold_scores = zeros(3, 1);
            
            for fold = 1:3
                % Get fold data
                X_train_fold = X_train(training(cv_partition, fold), :);
                y_train_fold = y_train(training(cv_partition, fold));
                X_val_fold = X_train(test(cv_partition, fold), :);
                y_val_fold = y_train(test(cv_partition, fold));
                
                % Train and evaluate
                model = TreeBagger(n_trees_grid(i), X_train_fold, y_train_fold, ...
                                 'Method', 'classification', ...
                                 'MaxNumSplits', max_depth_grid(j));
                [yhat, ~] = predict(model, X_val_fold);
                yhat = categorical(yhat);
                
                % Calculate F2 score
                cm = confusionmat(y_val_fold, yhat);
                precision = cm(2,2) / sum(cm(:,2));
                recall = cm(2,2) / sum(cm(2,:));
                beta = 2;
                f2 = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall);
                
                fold_scores(fold) = f2;
            end
            cv_scores(i,j) = mean(fold_scores);
        end
    end
    
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