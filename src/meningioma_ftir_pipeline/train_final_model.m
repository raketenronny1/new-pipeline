
%% PHASE 4: TRAIN FINAL MODEL ON ALL TRAINING DATA
% This script trains the final model using the best classifier from CV

function train_final_model(cfg)
    clear; clc;
    addpath('src/meningioma_ftir_pipeline');

    %% Load Best Classifier Info and Data
    fprintf('Loading best classifier info and data...\n');
    load(fullfile(cfg.paths.results, 'best_classifier_selection.mat'), 'best_model_info');
    load(fullfile(cfg.paths.results, 'X_train_pca.mat'), 'X_train_pca');
    load(fullfile(cfg.paths.results, 'preprocessed_data.mat'), 'trainingData');

% Set random seed
rng(42);

%% Train Final Model
fprintf('Training final model (%s)...\n', best_model_info.classifier);

switch best_model_info.classifier
    case 'LDA'
        final_model = fitcdiscr(X_train_pca, trainingData.y, ...
                              'DiscrimType', 'linear');
        
    case 'PLSDA'
        final_model = fitcpls(X_train_pca, trainingData.y, ...
                            'NumComponents', best_model_info.optimal_params.n_components);
        
    case 'SVM'
        final_model = fitcsvm(X_train_pca, trainingData.y, ...
                            'KernelFunction', 'rbf', ...
                            'BoxConstraint', best_model_info.optimal_params.C, ...
                            'KernelScale', 1/sqrt(best_model_info.optimal_params.gamma));
        
    case 'RandomForest'
        final_model = TreeBagger(best_model_info.optimal_params.n_trees, ...
                               X_train_pca, trainingData.y, ...
                               'Method', 'classification', ...
                               'MaxNumSplits', best_model_info.optimal_params.max_depth);
end

%% Package Final Model with Metadata
final_model_package = struct();
final_model_package.model = final_model;
final_model_package.classifier_type = best_model_info.classifier;
final_model_package.hyperparameters = best_model_info.optimal_params;
final_model_package.training_date = datestr(now);
final_model_package.n_training_samples = size(X_train_pca, 1);
final_model_package.cv_performance = best_model_info.cv_performance;

    %% Save Final Model
    save(fullfile(cfg.paths.models, 'final_model.mat'), 'final_model_package');

    fprintf('Final model training complete.\n');
end