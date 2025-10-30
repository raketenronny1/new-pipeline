classdef ClassifierWrapper < handle
    %CLASSIFIERWRAPPER Unified interface for multiple classifier types
    %
    % DESCRIPTION:
    %   Provides consistent train/predict interface across different
    %   classification algorithms. Handles model storage and prediction
    %   with posterior probabilities.
    %
    % SUPPORTED CLASSIFIERS:
    %   - PCA-LDA: PCA for dimensionality reduction + Linear Discriminant Analysis
    %   - SVM-RBF: Support Vector Machine with RBF kernel
    %   - PLS-DA: Partial Least Squares Discriminant Analysis
    %   - RandomForest: Random Forest ensemble classifier
    %
    % USAGE:
    %   % Create classifier
    %   cfg = Config.getInstance();
    %   clf = ClassifierWrapper('PCA-LDA', cfg);
    %   
    %   % Train
    %   clf.train(X_train, y_train);
    %   
    %   % Predict
    %   [y_pred, scores] = clf.predict(X_test);
    %
    % See also: Config, PreprocessingPipeline, CrossValidationEngine
    
    properties (Access = private)
        classifier_type  % Type of classifier
        config          % Configuration structure
        model           % Trained model object
        pca_model       % PCA model (for PCA-LDA only)
        classes         % Class labels
        verbose         % Display progress
    end
    
    methods
        function obj = ClassifierWrapper(classifier_type, config, varargin)
            %CLASSIFIERWRAPPER Constructor
            %
            % SYNTAX:
            %   clf = ClassifierWrapper('PCA-LDA', cfg)
            %   clf = ClassifierWrapper('SVM-RBF', cfg, 'Verbose', true)
            
            % Parse inputs
            p = inputParser;
            addRequired(p, 'classifier_type', @(x) ischar(x) || isstring(x));
            addRequired(p, 'config');
            addParameter(p, 'Verbose', false, @islogical);
            parse(p, classifier_type, config, varargin{:});
            
            obj.classifier_type = char(p.Results.classifier_type);
            obj.config = config;
            obj.verbose = p.Results.Verbose;
            
            % Validate classifier type
            valid_types = {'PCA-LDA', 'SVM-RBF', 'PLS-DA', 'RandomForest'};
            if ~ismember(obj.classifier_type, valid_types)
                error('ClassifierWrapper:InvalidType', ...
                    'Invalid classifier type: %s. Must be one of: %s', ...
                    obj.classifier_type, strjoin(valid_types, ', '));
            end
            
            if obj.verbose
                fprintf('[ClassifierWrapper] Created: %s\n', obj.classifier_type);
            end
        end
        
        function train(obj, X, y)
            %TRAIN Train classifier on data
            %
            % SYNTAX:
            %   clf.train(X_train, y_train)
            %
            % INPUTS:
            %   X: [n_samples × n_features] feature matrix
            %   y: Categorical labels
            
            % Validate inputs
            validateattributes(X, {'double'}, {'2d', 'finite'}, 'train', 'X');
            validateattributes(y, {'categorical'}, {'vector'}, 'train', 'y');
            
            if length(y) ~= size(X, 1)
                error('ClassifierWrapper:DimensionMismatch', ...
                    'Number of samples in X (%d) and y (%d) must match', ...
                    size(X, 1), length(y));
            end
            
            % Store classes
            obj.classes = unique(y);
            
            if obj.verbose
                fprintf('[%s] Training on %d samples, %d features\n', ...
                    obj.classifier_type, size(X, 1), size(X, 2));
            end
            
            % Train appropriate classifier
            switch obj.classifier_type
                case 'PCA-LDA'
                    obj.train_pca_lda(X, y);
                    
                case 'SVM-RBF'
                    obj.train_svm(X, y);
                    
                case 'PLS-DA'
                    obj.train_plsda(X, y);
                    
                case 'RandomForest'
                    obj.train_rf(X, y);
            end
            
            if obj.verbose
                fprintf('[%s] Training complete\n', obj.classifier_type);
            end
        end
        
        function [y_pred, scores] = predict(obj, X)
            %PREDICT Predict labels and scores for new data
            %
            % SYNTAX:
            %   [y_pred, scores] = clf.predict(X_test)
            %
            % OUTPUTS:
            %   y_pred: Predicted categorical labels
            %   scores: [n_samples × n_classes] posterior probabilities
            
            % Validate inputs
            validateattributes(X, {'double'}, {'2d', 'finite'}, 'predict', 'X');
            
            if isempty(obj.model)
                error('ClassifierWrapper:NotTrained', ...
                    'Classifier must be trained before prediction');
            end
            
            if obj.verbose
                fprintf('[%s] Predicting %d samples\n', ...
                    obj.classifier_type, size(X, 1));
            end
            
            % Predict with appropriate classifier
            switch obj.classifier_type
                case 'PCA-LDA'
                    [y_pred, scores] = obj.predict_pca_lda(X);
                    
                case 'SVM-RBF'
                    [y_pred, scores] = obj.predict_svm(X);
                    
                case 'PLS-DA'
                    [y_pred, scores] = obj.predict_plsda(X);
                    
                case 'RandomForest'
                    [y_pred, scores] = obj.predict_rf(X);
            end
        end
        
        function name = getName(obj)
            %GETNAME Get classifier name
            name = obj.classifier_type;
        end
        
        function mdl = getModel(obj)
            %GETMODEL Get trained model object
            mdl = obj.model;
        end
    end
    
    methods (Access = private)
        %% PCA-LDA
        function train_pca_lda(obj, X, y)
            %TRAIN_PCA_LDA Train PCA followed by LDA
            %
            % Based on MATLAB fitcdiscr documentation
            
            % Get PCA parameters from config
            if isstruct(obj.config)
                variance_threshold = obj.config.pca_variance_threshold;
                max_components = obj.config.pca_max_components;
            else
                variance_threshold = obj.config.get('pca_variance_threshold');
                max_components = obj.config.get('pca_max_components');
            end
            
            % Step 1: PCA for dimensionality reduction
            [coeff, score, ~, ~, explained] = pca(X, 'Economy', false);
            
            % Determine number of components
            cumvar = cumsum(explained);
            n_comp = find(cumvar >= variance_threshold * 100, 1, 'first');
            if isempty(n_comp)
                n_comp = length(explained);
            end
            n_comp = min(n_comp, max_components);
            n_comp = min(n_comp, size(X, 2));
            n_comp = max(n_comp, 1);  % At least 1 component
            
            % Store PCA model
            obj.pca_model = struct();
            obj.pca_model.coeff = coeff(:, 1:n_comp);
            obj.pca_model.mu = mean(X, 1);
            obj.pca_model.n_components = n_comp;
            obj.pca_model.explained = explained(1:n_comp);
            
            % Project data
            X_pca = score(:, 1:n_comp);
            
            if obj.verbose
                fprintf('  PCA: %d components (%.1f%% variance)\n', ...
                    n_comp, cumvar(n_comp));
            end
            
            % Step 2: LDA on PCA scores
            % Use fitcdiscr with linear discriminant type
            obj.model = fitcdiscr(X_pca, y, ...
                'DiscrimType', 'linear', ...
                'FillCoeffs', 'off');
        end
        
        function [y_pred, scores] = predict_pca_lda(obj, X)
            %PREDICT_PCA_LDA Predict with PCA-LDA
            %
            % Returns posterior probabilities as scores
            
            % Project to PCA space
            X_centered = X - obj.pca_model.mu;
            X_pca = X_centered * obj.pca_model.coeff;
            
            % Predict with LDA (returns labels and posterior probabilities)
            [y_pred, scores] = predict(obj.model, X_pca);
            y_pred = categorical(y_pred);
        end
        
        %% SVM-RBF
        function train_svm(obj, X, y)
            %TRAIN_SVM Train SVM with RBF kernel
            %
            % Based on MATLAB fitcsvm documentation
            
            % Get SVM parameters from config
            if isstruct(obj.config)
                C = obj.config.svm_C;
                kernel_scale = obj.config.svm_kernel_scale;
            else
                C = obj.config.get('svm_C');
                kernel_scale = obj.config.get('svm_kernel_scale');
            end
            
            % Train SVM with proper parameters
            if strcmp(kernel_scale, 'auto')
                % Auto kernel scale
                obj.model = fitcsvm(X, y, ...
                    'KernelFunction', 'rbf', ...
                    'BoxConstraint', C, ...
                    'KernelScale', 'auto', ...
                    'Standardize', false, ...
                    'ClassNames', unique(y));  % Ensure class order
            else
                obj.model = fitcsvm(X, y, ...
                    'KernelFunction', 'rbf', ...
                    'BoxConstraint', C, ...
                    'KernelScale', kernel_scale, ...
                    'Standardize', false, ...
                    'ClassNames', unique(y));
            end
            
            if obj.verbose
                fprintf('  SVM: C=%.2f, KernelScale=%s\n', C, string(kernel_scale));
            end
        end
        
        function [y_pred, scores] = predict_svm(obj, X)
            %PREDICT_SVM Predict with SVM
            %
            % For binary SVM, fitcsvm returns decision values
            % We need to convert to posterior probabilities
            
            % Get predictions and decision scores
            [y_pred, decision_scores] = predict(obj.model, X);
            y_pred = categorical(y_pred);
            
            % Convert decision values to posterior probabilities
            % For binary classification, decision_scores is [n × 1]
            % Positive values favor positive class, negative favor negative
            
            if size(decision_scores, 2) == 1
                % Binary classification: use sigmoid/logistic function
                % P(class=positive) = 1 / (1 + exp(-decision_score))
                prob_pos = 1 ./ (1 + exp(-decision_scores));
                prob_neg = 1 - prob_pos;
                
                % Order columns: [negative_class, positive_class]
                scores = [prob_neg, prob_pos];
            else
                % Multi-class: decision scores already provided
                % Convert to probabilities using softmax
                scores = exp(decision_scores);
                scores = scores ./ sum(scores, 2);
            end
        end
        
        %% PLS-DA
        function train_plsda(obj, X, y)
            %TRAIN_PLSDA Train PLS-DA
            
            % Get PLS parameters from config
            if isstruct(obj.config)
                n_comp = obj.config.pls_ncomp;
            else
                n_comp = obj.config.get('pls_ncomp');
            end
            
            % Limit components to min(n_comp, n_features, n_samples-1)
            n_comp = min([n_comp, size(X, 2), size(X, 1) - 1]);
            
            % Convert categorical labels to numeric
            [y_numeric, class_names] = grp2idx(y);
            
            % Train PLS regression
            [~, ~, ~, ~, BETA] = plsregress(X, y_numeric, n_comp);
            
            % Store model
            obj.model = struct();
            obj.model.BETA = BETA;
            obj.model.n_components = n_comp;
            obj.model.class_names = class_names;
            
            if obj.verbose
                fprintf('  PLS-DA: %d components\n', n_comp);
            end
        end
        
        function [y_pred, scores] = predict_plsda(obj, X)
            %PREDICT_PLSDA Predict with PLS-DA
            
            % Predict continuous values
            y_fit = [ones(size(X, 1), 1), X] * obj.model.BETA;
            
            % For binary classification (grp2idx returns 1, 2)
            % y_fit predicts values around 1 or 2
            % We need to convert to class assignments
            
            % Get predicted class (round to nearest integer class)
            y_pred_idx = round(y_fit);
            y_pred_idx = max(1, min(y_pred_idx, length(obj.model.class_names)));
            
            % Convert to categorical
            y_pred = categorical(obj.model.class_names(y_pred_idx));
            
            % Compute scores based on distance from class centers
            n_classes = length(obj.model.class_names);
            scores = zeros(size(X, 1), n_classes);
            
            for i = 1:n_classes
                % Distance from class center (closer = higher score)
                dist = abs(y_fit - i);
                scores(:, i) = exp(-dist);  % Exponential decay with distance
            end
            
            % Normalize scores to sum to 1 (probabilities)
            scores = scores ./ sum(scores, 2);
        end
        
        %% Random Forest
        function train_rf(obj, X, y)
            %TRAIN_RF Train Random Forest
            %
            % Based on MATLAB fitcensemble documentation
            
            % Get RF parameters from config
            if isstruct(obj.config)
                n_trees = obj.config.rf_ntrees;
                min_leaf = obj.config.rf_min_leaf_size;
            else
                n_trees = obj.config.get('rf_ntrees');
                min_leaf = obj.config.get('rf_min_leaf_size');
            end
            
            % Create decision tree template
            tree_template = templateTree('MinLeafSize', min_leaf, ...
                                         'MaxNumSplits', size(X, 2));
            
            % Train Random Forest using fitcensemble
            obj.model = fitcensemble(X, y, ...
                'Method', 'Bag', ...
                'NumLearningCycles', n_trees, ...
                'Learners', tree_template, ...
                'Type', 'classification');
            
            if obj.verbose
                fprintf('  RandomForest: %d trees, MinLeaf=%d\n', ...
                    n_trees, min_leaf);
            end
        end
        
        function [y_pred, scores] = predict_rf(obj, X)
            %PREDICT_RF Predict with Random Forest
            %
            % Returns posterior probabilities from ensemble
            
            % Predict returns labels and posterior probabilities
            [y_pred, scores] = predict(obj.model, X);
            y_pred = categorical(y_pred);
        end
    end
end
