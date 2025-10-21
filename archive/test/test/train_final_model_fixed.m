%% PHASE 4: TRAIN FINAL MODEL ON ALL TRAINING DATA
% This script trains the final model using the best classifier from CV
% Fixed version that handles missing optimal_params field

function final_model = train_final_model_fixed(cfg, cv_results)
    % Input validation
    if ~isstruct(cfg) || ~isfield(cfg, 'paths') || ~isfield(cfg.paths, 'results') || ~isfield(cfg.paths, 'models')
        error('Invalid cfg structure. Must contain paths.results and paths.models');
    end
    
    % cv_results is optional
    if nargin < 2 || isempty(cv_results)
        fprintf('cv_results not provided, will load from file\n');
        cv_results = [];
    end

    %% Load Best Classifier Info and Data
    fprintf('Loading best classifier info and data...\n');
        
        % Load PCA-transformed data and original data
        load(fullfile(cfg.paths.results, 'X_train_pca.mat'), 'X_train_pca');
        load(fullfile(cfg.paths.results, 'preprocessed_data.mat'), 'trainingData');
        
        % If cv_results was provided, use it; otherwise load from file
        if isempty(cv_results)
            load(fullfile(cfg.paths.results, 'best_classifier_selection.mat'), 'best_model_info');
        else
            % Find the best classifier from cv_results
            fprintf('Using provided cv_results to determine best classifier...\n');
            
            % Extract performance metrics
            classifier_names = {'LDA', 'PLSDA', 'SVM', 'RandomForest'};
            n_classifiers = length(cv_results);
            mean_f2_scores = zeros(n_classifiers, 1);
            
            for i = 1:n_classifiers
                if isfield(cv_results{i}, 'performance') && ~isempty(cv_results{i}.performance)
                    % Extract F2 scores
                    if iscell(cv_results{i}.performance)
                        all_f2 = [];
                        for j = 1:length(cv_results{i}.performance)
                            if ~isempty(cv_results{i}.performance{j}) && isfield(cv_results{i}.performance{j}, 'f2')
                                all_f2 = [all_f2, cv_results{i}.performance{j}.f2];
                            end
                        end
                        mean_f2_scores(i) = mean(all_f2);
                    else
                        if isfield(cv_results{i}.performance, 'f2')
                            all_f2 = [cv_results{i}.performance.f2];
                            mean_f2_scores(i) = mean(all_f2);
                        end
                    end
                end
            end
            
            % Find the best classifier
            [~, best_idx] = max(mean_f2_scores);
            best_classifier = classifier_names{best_idx};
            
            % Create best_model_info structure
            best_model_info = struct();
            best_model_info.classifier = best_classifier;
            
            % Check for optimal_params field and provide defaults if missing
            if isfield(cv_results{best_idx}, 'optimal_params')
                best_model_info.optimal_params = cv_results{best_idx}.optimal_params;
            else
                fprintf('Warning: optimal_params field missing. Using default parameters.\n');
                % Create default parameters based on classifier type
                switch best_classifier
                    case 'LDA'
                        best_model_info.optimal_params = struct();
                    case 'PLSDA'
                        best_model_info.optimal_params = struct('n_components', 3);
                        fprintf('Using default PLSDA components: 3\n');
                    case 'SVM'
                        best_model_info.optimal_params = struct('C', 1.0, 'gamma', 0.1);
                        fprintf('Using default SVM parameters: C=1.0, gamma=0.1\n');
                    case 'RandomForest'
                        best_model_info.optimal_params = struct('n_trees', 100, 'max_depth', 10);
                        fprintf('Using default Random Forest parameters: n_trees=100, max_depth=10\n');
                end
            end
            
            fprintf('Selected %s as best classifier (F2 score: %.3f)\n', ...
                    best_classifier, mean_f2_scores(best_idx));
        end

% Set random seed from config or use default
if isfield(cfg, 'random_seed')
    rng(cfg.random_seed, 'twister');
else
    rng(42, 'twister');
    warning('No random seed specified in cfg. Using default seed 42.');
end

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
final_model_package.training_date = string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
final_model_package.n_training_samples = size(X_train_pca, 1);

% Create a placeholder for cv_performance if not available
if isfield(best_model_info, 'cv_performance')
    final_model_package.cv_performance = best_model_info.cv_performance;
else
    final_model_package.cv_performance = struct('info', 'Performance metrics not available');
end

    %% Save Final Model
    save(fullfile(cfg.paths.models, 'final_model.mat'), 'final_model_package');
    fprintf('Final model training complete.\n');
end