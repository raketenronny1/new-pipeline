%% PATIENT-WISE CROSS-VALIDATION WITH MAJORITY VOTING
% Main cross-validation runner that:
% 1. Performs patient-wise stratified K-fold CV
% 2. Trains on individual spectra
% 3. Predicts each spectrum individually
% 4. Aggregates to patient-level via majority voting
%
% Following best practices from:
% - Baker et al. (2014) Nature Protocols 9(8):1771-1791
% - Greener et al. (2022) Nature Reviews Molecular Cell Biology 23:40-55

function [cvResults] = run_patientwise_cross_validation(cfg)
    % Runs K-fold patient-wise stratified cross-validation
    %
    % INPUT:
    %   cfg: configuration struct with paths and parameters
    %
    % OUTPUT:
    %   cvResults: struct with detailed results per fold and aggregated metrics
    
    %% Load Patient-Wise Data
    fprintf('\n=== PATIENT-WISE CROSS-VALIDATION ===\n');
    fprintf('Loading patient-wise data...\n');
    
    data_file = fullfile(cfg.paths.results, 'patientwise_data.mat');
    if ~exist(data_file, 'file')
        error('Patient-wise data not found. Run load_and_prepare_data_patientwise first.');
    end
    load(data_file, 'trainingData', 'wavenumbers_roi');
    
    patientData = trainingData.patientData;
    fprintf('Loaded %d patients for cross-validation\n', length(patientData));
    
    %% Load PCA Model (if feature selection was performed)
    pca_file = fullfile(cfg.paths.models, 'pca_model.mat');
    use_pca = false;
    if exist(pca_file, 'file')
        load(pca_file, 'pca_model');
        use_pca = true;
        fprintf('PCA model loaded - will apply dimensionality reduction\n');
    else
        fprintf('No PCA model found - using full spectra\n');
    end
    
    %% CV Parameters
    if isfield(cfg, 'cv') && isfield(cfg.cv, 'n_folds')
        K = cfg.cv.n_folds;
    else
        K = 5;
    end
    
    if isfield(cfg, 'random_seed')
        random_seed = cfg.random_seed;
    else
        random_seed = 42;
    end
    
    % Set classifier type
    if isfield(cfg, 'classifiers') && isfield(cfg.classifiers, 'primary_type')
        classifier_type = cfg.classifiers.primary_type;
    else
        classifier_type = 'SVM';  % Default
    end
    
    fprintf('\nCV Configuration:\n');
    fprintf('  Folds: %d\n', K);
    fprintf('  Classifier: %s\n', classifier_type);
    fprintf('  Random seed: %d\n', random_seed);
    fprintf('  Feature reduction: %s\n', mat2str(use_pca));
    
    %% Create CV Folds (Patient-Wise)
    addpath(fileparts(mfilename('fullpath')));  % Add path for helper functions
    cvFolds = createPatientWiseStratifiedCV(patientData, K, random_seed);
    
    %% Initialize Results Storage
    cvResults = struct('fold', {}, ...
                       'spectrumLevelResults', {}, ...
                       'patientLevelResults', {}, ...
                       'trainedModel', {}, ...
                       'preprocessing', {}, ...
                       'spectrumMetrics', {}, ...
                       'patientMetrics', {}, ...
                       'confidenceMetrics', {});
    
    %% Cross-Validation Loop
    for k = 1:K
        fprintf('\n========== FOLD %d/%d ==========\n', k, K);
        
        % Extract data for this fold
        [X_train, y_train, X_test, y_test, testPatientIDs, testSpectrumMap] = ...
            extractSpectraForFold(patientData, ...
                                  cvFolds(k).trainPatientIdx, ...
                                  cvFolds(k).testPatientIdx);
        
        % Preprocess (normalization, scaling, etc.)
        [X_train_prep, prepParams] = preprocessSpectra(X_train, cfg);
        X_test_prep = applyPreprocessing(X_test, prepParams);
        
        % Apply PCA if available
        if use_pca
            X_train_prep = X_train_prep * pca_model.coeff(:, 1:pca_model.n_components);
            X_test_prep = X_test_prep * pca_model.coeff(:, 1:pca_model.n_components);
            fprintf('  Applied PCA: %d features -> %d components\n', ...
                    size(X_train,2), pca_model.n_components);
        end
        
        % Train classifier
        fprintf('  Training %s classifier...\n', classifier_type);
        [model, train_time] = trainClassifier(X_train_prep, y_train, classifier_type, cfg);
        fprintf('  Training time: %.2f seconds\n', train_time);
        
        % Predict on ALL test spectra individually
        fprintf('  Predicting %d test spectra...\n', length(y_test));
        [y_pred_spectra, y_pred_prob, pred_time] = predictSpectra(model, X_test_prep, classifier_type);
        fprintf('  Prediction time: %.2f seconds\n', pred_time);
        
        % Aggregate predictions per patient (MAJORITY VOTE)
        fprintf('  Aggregating predictions to patient-level...\n');
        [patientPredictions, ~] = aggregatePredictionsPerPatient(...
            y_pred_spectra, y_pred_prob, testPatientIDs, patientData, cvFolds(k).testPatientIdx);
        
        % Store results
        cvResults(k).fold = k;
        cvResults(k).spectrumLevelResults = struct(...
            'y_true', y_test, ...
            'y_pred', y_pred_spectra, ...
            'y_prob', y_pred_prob, ...
            'patientIDs', {testPatientIDs}, ...
            'spectrumMap', testSpectrumMap);
        
        cvResults(k).patientLevelResults = patientPredictions;
        cvResults(k).trainedModel = model;
        cvResults(k).preprocessing = prepParams;
        cvResults(k).trainTime = train_time;
        cvResults(k).predTime = pred_time;
        
        % Compute metrics
        cvResults(k) = computeMetrics(cvResults(k));
        
        % Display fold results
        displayFoldResults(cvResults(k));
    end
    
    % Aggregate results across all folds
    cvResults = aggregateCVResults(cvResults);
    
    %% Save Results
    fprintf('\nSaving cross-validation results...\n');
    save(fullfile(cfg.paths.results, 'cv_results_patientwise.mat'), 'cvResults', '-v7.3');
    
    fprintf('✓ Patient-wise cross-validation complete!\n');
end


%% PREPROCESSING FUNCTION
function [X_prep, prepParams] = preprocessSpectra(X, cfg)
    % Preprocesses spectra with normalization and scaling
    %
    % This should match the preprocessing used in the original pipeline
    
    prepParams = struct();
    
    % Vector normalization (L2 norm)
    X_norm = X ./ vecnorm(X, 2, 2);
    prepParams.normalization = 'L2';
    
    % Standardization (zero mean, unit variance)
    prepParams.mean = mean(X_norm, 1);
    prepParams.std = std(X_norm, 0, 1);
    prepParams.std(prepParams.std == 0) = 1;  % Avoid division by zero
    
    X_prep = (X_norm - prepParams.mean) ./ prepParams.std;
    
    prepParams.method = 'L2_norm_then_standardize';
end


function [X_prep] = applyPreprocessing(X, prepParams)
    % Applies preprocessing parameters from training to new data
    
    % Vector normalization
    X_norm = X ./ vecnorm(X, 2, 2);
    
    % Apply training standardization parameters
    X_prep = (X_norm - prepParams.mean) ./ prepParams.std;
end


%% TRAINING FUNCTION
function [model, train_time] = trainClassifier(X_train, y_train, classifier_type, cfg)
    % Trains a classifier on the training data
    
    tic;
    
    switch classifier_type
        case 'SVM'
            % Use default or configured hyperparameters
            if isfield(cfg, 'classifiers') && isfield(cfg.classifiers, 'svm_C')
                C = cfg.classifiers.svm_C;
            else
                C = 1;
            end
            
            if isfield(cfg, 'classifiers') && isfield(cfg.classifiers, 'svm_gamma')
                gamma = cfg.classifiers.svm_gamma;
            else
                gamma = 1 / size(X_train, 2);  % Auto-scale
            end
            
            model = fitcsvm(X_train, y_train, ...
                          'KernelFunction', 'rbf', ...
                          'BoxConstraint', C, ...
                          'KernelScale', 1/sqrt(gamma), ...
                          'Standardize', false);  % Already standardized
            
            % Fit posterior for probability estimates
            model = fitPosterior(model);
            
        case 'LDA'
            model = fitcdiscr(X_train, y_train, 'DiscrimType', 'linear');
            
        case 'RandomForest'
            if isfield(cfg, 'classifiers') && isfield(cfg.classifiers, 'rf_n_trees')
                n_trees = cfg.classifiers.rf_n_trees;
            else
                n_trees = 100;
            end
            
            model = TreeBagger(n_trees, X_train, y_train, ...
                             'Method', 'classification', ...
                             'OOBPrediction', 'on', ...
                             'MinLeafSize', 5);
            
        otherwise
            error('Unsupported classifier type: %s', classifier_type);
    end
    
    train_time = toc;
end


%% PREDICTION FUNCTION
function [y_pred, y_prob, pred_time] = predictSpectra(model, X_test, classifier_type)
    % Predicts labels and probabilities for test spectra
    
    tic;
    
    switch classifier_type
        case {'SVM', 'LDA'}
            [y_pred, scores] = predict(model, X_test);
            
            % Convert scores to probabilities if needed
            if size(scores, 2) == 2
                y_prob = scores;
            else
                % Single score - convert to 2-class probabilities
                y_prob = zeros(length(y_pred), 2);
                y_prob(:, 1) = 1 ./ (1 + exp(scores));  % WHO-1
                y_prob(:, 2) = 1 ./ (1 + exp(-scores)); % WHO-3
                % Normalize
                y_prob = y_prob ./ sum(y_prob, 2);
            end
            
        case 'RandomForest'
            [y_pred_cell, scores] = predict(model, X_test);
            % Convert cell array to numeric
            y_pred = cellfun(@str2double, y_pred_cell);
            y_prob = scores;
            
        otherwise
            error('Unsupported classifier type: %s', classifier_type);
    end
    
    pred_time = toc;
end
