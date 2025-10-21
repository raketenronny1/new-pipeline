
%% PHASE 5: TEST SET EVALUATION (DONE EXACTLY ONCE)
% This script evaluates the final model on the held-out test set

function evaluate_test_set(cfg)
        %% Load Models and Test Data
        fprintf('Loading models and test data...\n');
        load(fullfile(cfg.paths.models, 'pca_model.mat'), 'pca_model');
        load(fullfile(cfg.paths.models, 'final_model.mat'), 'final_model_package');
        load(fullfile(cfg.paths.results, 'preprocessed_data.mat'), 'testData');

%% Transform Test Data
fprintf('Transforming test data...\n');

% Center test data using TRAINING mean
X_test_centered = testData.X - pca_model.mu;

% Project using TRAINING PC loadings
X_test_pca = X_test_centered * pca_model.coeff(:, 1:pca_model.n_components);

fprintf('Test set transformed: %d samples × %d PCs\n', size(X_test_pca));

%% Make Predictions
fprintf('Making predictions...\n');

% Get predictions and probability scores
[y_pred, scores] = predict(final_model_package.model, X_test_pca);

% For Random Forest, handle cell array output
if strcmp(final_model_package.classifier_type, 'RandomForest')
    y_pred = categorical(y_pred);
end

%% Calculate Performance Metrics
fprintf('Calculating performance metrics...\n');

% Confusion matrix
cm = confusionmat(testData.y, y_pred);

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
if strcmp(final_model_package.classifier_type, 'RandomForest')
    [X_roc, Y_roc, T, AUC] = perfcurve(testData.y, str2double(scores(:,2)), 'WHO-3');
else
    [X_roc, Y_roc, T, AUC] = perfcurve(testData.y, scores(:,2), 'WHO-3');
end

%% Create Visualizations

% --- Confusion Matrix Heatmap ---
figure('Position', [100, 100, 800, 600]);
h = heatmap({'Predicted WHO-1', 'Predicted WHO-3'}, ...
            {'True WHO-1', 'True WHO-3'}, cm, ...
            'Colormap', parula, 'ColorbarVisible', 'on');
h.Title = sprintf('Test Set Confusion Matrix (n=%d)', size(X_test_pca, 1));
h.XLabel = 'Predicted Class';
h.YLabel = 'True Class';
        saveas(gcf, fullfile(cfg.paths.results, 'test_confusion_matrix.png'));

% --- ROC Curve ---
figure('Position', [100, 100, 700, 600]);
plot(X_roc, Y_roc, 'b-', 'LineWidth', 2);
hold on;
plot([0, 1], [0, 1], 'k--', 'LineWidth', 1);  % Chance line
xlabel('False Positive Rate (1-Specificity)');
ylabel('True Positive Rate (Sensitivity)');
title(sprintf('ROC Curve (AUC = %.3f)', AUC));
legend(sprintf('%s (AUC=%.3f)', final_model_package.classifier_type, AUC), ...
       'Chance', 'Location', 'southeast');
grid on;
        saveas(gcf, fullfile(cfg.paths.results, 'test_roc_curve.png'));

% --- PCA Scatter Plot with Predictions ---
figure('Position', [100, 100, 900, 600]);

% Correct predictions
correct_idx = (y_pred == testData.y);
scatter(X_test_pca(correct_idx & testData.y=='WHO-1', 1), ...
        X_test_pca(correct_idx & testData.y=='WHO-1', 2), ...
        100, 'b', 'filled', 'MarkerFaceAlpha', 0.7);
hold on;
scatter(X_test_pca(correct_idx & testData.y=='WHO-3', 1), ...
        X_test_pca(correct_idx & testData.y=='WHO-3', 2), ...
        100, 'r', 'filled', 'MarkerFaceAlpha', 0.7);

% Incorrect predictions (marked with X)
incorrect_idx = ~correct_idx;
scatter(X_test_pca(incorrect_idx & testData.y=='WHO-1', 1), ...
        X_test_pca(incorrect_idx & testData.y=='WHO-1', 2), ...
        150, 'b', 'x', 'LineWidth', 3);
scatter(X_test_pca(incorrect_idx & testData.y=='WHO-3', 1), ...
        X_test_pca(incorrect_idx & testData.y=='WHO-3', 2), ...
        150, 'r', 'x', 'LineWidth', 3);

xlabel('PC1'); ylabel('PC2');
legend({'WHO-1 (Correct)', 'WHO-3 (Correct)', ...
        'WHO-1 (Misclassified)', 'WHO-3 (Misclassified)'}, ...
        'Location', 'best');
title('Test Set Predictions in PCA Space');
grid on;
        saveas(gcf, fullfile(cfg.paths.results, 'test_pca_predictions.png'));

%% Save Results
test_results = struct();
test_results.predictions = y_pred;
test_results.scores = scores;
test_results.true_labels = testData.y;
test_results.probe_ids = testData.probe_ids;

% Performance metrics
test_results.metrics.accuracy = accuracy;
test_results.metrics.balanced_accuracy = balanced_accuracy;
test_results.metrics.sensitivity_WHO3 = sensitivity_WHO3;
test_results.metrics.specificity_WHO1 = specificity_WHO1;
test_results.metrics.ppv = ppv;
test_results.metrics.npv = npv;
test_results.metrics.f1 = f1;
test_results.metrics.f2 = f2;
test_results.metrics.auc = AUC;
test_results.metrics.confusion_matrix = cm;

% Misclassified samples
misclassified_idx = find(y_pred ~= testData.y);
test_results.misclassified_samples = testData.probe_ids(misclassified_idx);

        save(fullfile(cfg.paths.results, 'test_results.mat'), 'test_results');

% Create summary table
summary_table = table();
summary_table.Metric = {'Accuracy'; 'Balanced_Accuracy'; 'Sensitivity_WHO3'; ...
                       'Specificity_WHO1'; 'PPV'; 'NPV'; 'F1_Score'; ...
                       'F2_Score'; 'AUC_ROC'};
summary_table.Value = [accuracy; balanced_accuracy; sensitivity_WHO3; ...
                      specificity_WHO1; ppv; npv; f1; f2; AUC];
        writetable(summary_table, fullfile(cfg.paths.results, 'test_performance.csv'));

        fprintf('\nTest set evaluation complete.\n');
end