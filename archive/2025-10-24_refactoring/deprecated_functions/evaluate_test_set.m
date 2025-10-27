%% PHASE 5: TEST SET EVALUATION (DONE EXACTLY ONCE)
% This script evaluates the final model on the held-out test set

function test_results = evaluate_test_set(cfg, final_model_package)
    %% Load Models and Test Data
    fprintf('Loading models and test data...\n');
    
    % Load data - either from parameters or from files
    if nargin < 2
        load(fullfile(cfg.paths.models, 'final_model.mat'), 'final_model_package');
    end
    
    load(fullfile(cfg.paths.models, 'pca_model.mat'), 'pca_model');
    load(fullfile(cfg.paths.results, 'preprocessed_data.mat'), 'testData');

    %% Transform Test Data
    fprintf('Transforming test data...\n');

    % Center test data using TRAINING mean
    X_test_centered = testData.X - pca_model.mu;

    % Project using TRAINING PC loadings
    X_test_pca = X_test_centered * pca_model.coeff(:, 1:pca_model.n_components);

    fprintf('Test set transformed: %d samples × %d PCs\n', size(X_test_pca, 1), size(X_test_pca, 2));

    %% Make Predictions
    fprintf('Making predictions...\n');

    % Get predictions and probability scores
    % Check if we have a model directly or a package structure
    if isa(final_model_package, 'struct') && isfield(final_model_package, 'model')
        % We have a package structure with a model field
        model = final_model_package.model;
        classifier_type = final_model_package.classifier_type;
    else
        % The model was passed directly
        model = final_model_package;
        % Try to determine the classifier type from the model
        if isa(model, 'TreeBagger')
            classifier_type = 'RandomForest';
        elseif isa(model, 'ClassificationDiscriminant')
            classifier_type = 'LDA';
        elseif isa(model, 'ClassificationSVM')
            classifier_type = 'SVM';
        elseif isa(model, 'ClassificationPLS')
            classifier_type = 'PLSDA';
        else
            classifier_type = 'Unknown';
        end
    end
    
    % Get predictions using the model
    [y_pred, scores] = predict(model, X_test_pca);

    % For Random Forest, handle cell array output
    if strcmp(classifier_type, 'RandomForest')
        if iscell(y_pred)
            y_pred = cellfun(@str2double, y_pred);
        end
    end

    %% Calculate Performance Metrics
    fprintf('Calculating performance metrics...\n');

    % Confusion matrix
    cm = confusionmat(testData.y, y_pred);
    
    % Display confusion matrix in text format
    fprintf('\nConfusion Matrix:\n');
    fprintf('            Predicted WHO-1  Predicted WHO-3\n');
    fprintf('True WHO-1      %6d           %6d\n', cm(1,1), cm(1,2));
    fprintf('True WHO-3      %6d           %6d\n', cm(2,1), cm(2,2));
    fprintf('\n');

    % Basic metrics
    accuracy = sum(diag(cm)) / sum(cm(:));
    balanced_accuracy = mean([cm(1,1)/sum(cm(1,:)), cm(2,2)/sum(cm(2,:))]);

    % Class-specific metrics
    sensitivity_WHO3 = cm(2,2) / sum(cm(2,:));  % True positive rate
    specificity_WHO1 = cm(1,1) / sum(cm(1,:));  % True negative rate

    % Predictive values
    ppv = cm(2,2) / sum(cm(:,2));  % Positive predictive value
    npv = cm(1,1) / sum(cm(:,1));  % Negative predictive value

    % F-scores
    precision = ppv;
    recall = sensitivity_WHO3;
    f1 = 2 * (precision * recall) / (precision + recall);

    % F2-score (emphasizes recall)
    beta = 2;
    f2 = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall);

    % ROC analysis
    try
        if strcmp(classifier_type, 'RandomForest')
            [~, ~, ~, AUC] = perfcurve(testData.y, str2double(scores(:,2)), 3); % WHO-3
        else
            [~, ~, ~, AUC] = perfcurve(testData.y, scores(:,2), 3); % WHO-3
        end
    catch ME
        warning('AUC:CalculationError', 'Could not calculate AUC: %s. Setting to NaN.', ME.message);
        AUC = NaN;
    end

    %% Skip Visualizations in Batch Mode
    fprintf('INFO: Skipping figure generation in batch mode\n');
    fprintf('INFO: Would create confusion matrix heatmap\n');
    fprintf('INFO: Would create ROC curve (AUC = %.3f)\n', AUC);
    fprintf('INFO: Would create PCA scatter plot with predictions\n');

    %% Output Performance Summary
    fprintf('\n=== TEST SET PERFORMANCE SUMMARY ===\n');
    fprintf('Classifier: %s\n', classifier_type);
    fprintf('Accuracy: %.2f%%\n', accuracy * 100);
    fprintf('Balanced Accuracy: %.2f%%\n', balanced_accuracy * 100);
    fprintf('Sensitivity (WHO-3): %.2f%%\n', sensitivity_WHO3 * 100);
    fprintf('Specificity (WHO-1): %.2f%%\n', specificity_WHO1 * 100);
    fprintf('Precision (PPV): %.2f%%\n', ppv * 100);
    fprintf('F1 Score: %.3f\n', f1);
    fprintf('F2 Score: %.3f\n', f2);
    fprintf('AUC-ROC: %.3f\n', AUC);
    fprintf('===================================\n');

    %% Save Results
    test_results = struct();
    test_results.confusion_matrix = cm;
    test_results.accuracy = accuracy;
    test_results.balanced_accuracy = balanced_accuracy;
    test_results.sensitivity = sensitivity_WHO3;
    test_results.specificity = specificity_WHO1;
    test_results.ppv = ppv;
    test_results.npv = npv;
    test_results.f1 = f1;
    test_results.f2 = f2;
    test_results.auc = AUC;
    test_results.predictions = y_pred;
    test_results.true_labels = testData.y;
    test_results.scores = scores;
    
    save(fullfile(cfg.paths.results, 'test_results.mat'), 'test_results');
end