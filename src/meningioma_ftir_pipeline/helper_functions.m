%% Helper Functions for Meningioma Classification Pipeline

function metrics = calculate_classification_metrics(y_true, y_pred, scores)
    % CALCULATE_CLASSIFICATION_METRICS - Calculate comprehensive classification metrics
    %
    % Inputs:
    %   y_true  - True class labels (categorical or string array)
    %   y_pred  - Predicted class labels
    %   scores  - Prediction scores/probabilities [n_samples × n_classes]
    %
    % Outputs:
    %   metrics - Structure containing all classification metrics
    
    % Confusion matrix
    cm = confusionmat(y_true, y_pred);
    
    % Basic metrics
    metrics.accuracy = sum(diag(cm)) / sum(cm(:));
    metrics.balanced_accuracy = mean([cm(1,1)/sum(cm(1,:)), cm(2,2)/sum(cm(2,:))]);
    
    % Class-specific metrics
    metrics.sensitivity = cm(2,2) / sum(cm(2,:));  % WHO-3 recall
    metrics.specificity = cm(1,1) / sum(cm(1,:));  % WHO-1 specificity
    
    % Predictive values
    metrics.ppv = cm(2,2) / sum(cm(:,2));  % Positive predictive value
    metrics.npv = cm(1,1) / sum(cm(:,1));  % Negative predictive value
    
    % F-scores
    metrics.f1 = 2 * (metrics.ppv * metrics.sensitivity) / ...
                (metrics.ppv + metrics.sensitivity);
    
    % F2-score (emphasizes recall)
    beta = 2;
    metrics.f2 = (1 + beta^2) * (metrics.ppv * metrics.sensitivity) / ...
                (beta^2 * metrics.ppv + metrics.sensitivity);
    
    % ROC analysis
    [~, ~, ~, metrics.auc] = perfcurve(y_true, scores(:,2), 'WHO-3');
    
    % Store confusion matrix
    metrics.confusion_matrix = cm;
end

function fig = plot_confusion_matrix(cm, class_names, title_str)
    % PLOT_CONFUSION_MATRIX - Create publication-quality confusion matrix figure
    %
    % Inputs:
    %   cm          - 2×2 confusion matrix
    %   class_names - Cell array of class names
    %   title_str   - Title for the plot
    %
    % Outputs:
    %   fig - Handle to the figure
    
    fig = figure('Position', [100, 100, 800, 600]);
    h = heatmap(class_names, class_names, cm, ...
                'Colormap', parula, 'ColorbarVisible', 'on');
    h.Title = title_str;
    h.XLabel = 'Predicted Class';
    h.YLabel = 'True Class';
    
    % Add percentages in cells
    h.CellLabelFormat = '%g\n(%.1f%%)';
    for i = 1:size(cm,1)
        for j = 1:size(cm,2)
            h.CellLabelData(i,j) = sprintf('%d\n(%.1f%%)', ...
                                  cm(i,j), 100*cm(i,j)/sum(cm(i,:)));
        end
    end
end

function fig = plot_roc_curve_with_ci(y_true, scores, n_bootstrap)
    % PLOT_ROC_CURVE_WITH_CI - Create ROC curve with confidence intervals
    %
    % Inputs:
    %   y_true      - True class labels
    %   scores      - Prediction scores/probabilities
    %   n_bootstrap - Number of bootstrap iterations
    %
    % Outputs:
    %   fig - Handle to the figure
    
    % Calculate main ROC curve
    [X, Y, T, AUC] = perfcurve(y_true, scores(:,2), 'WHO-3');
    
    % Bootstrap for confidence intervals
    n_samples = length(y_true);
    auc_boot = zeros(n_bootstrap, 1);
    
    for i = 1:n_bootstrap
        % Bootstrap sampling
        boot_idx = randsample(n_samples, n_samples, true);
        y_boot = y_true(boot_idx);
        scores_boot = scores(boot_idx,:);
        
        % Calculate AUC for this bootstrap sample
        [~, ~, ~, auc_boot(i)] = perfcurve(y_boot, scores_boot(:,2), 'WHO-3');
    end
    
    % Calculate confidence intervals
    ci = prctile(auc_boot, [2.5 97.5]);
    
    % Create plot
    fig = figure('Position', [100, 100, 700, 600]);
    plot(X, Y, 'b-', 'LineWidth', 2);
    hold on;
    plot([0, 1], [0, 1], 'k--', 'LineWidth', 1);  % Chance line
    xlabel('False Positive Rate (1-Specificity)');
    ylabel('True Positive Rate (Sensitivity)');
    title(sprintf('ROC Curve (AUC = %.3f, 95%% CI: %.3f-%.3f)', ...
                 AUC, ci(1), ci(2)));
    grid on;
end

function outlier_mask = detect_mahalanobis_outliers(data, confidence)
    % DETECT_MAHALANOBIS_OUTLIERS - Detect outliers using Mahalanobis distance
    %
    % Inputs:
    %   data       - Data matrix [n_samples × n_features]
    %   confidence - Confidence level for chi-squared threshold
    %
    % Outputs:
    %   outlier_mask - Logical array indicating outliers
    
    % Compute PCA to handle potential collinearity
    [coeff, score] = pca(data);
    
    % Use first few PCs for outlier detection
    n_pcs = min(10, size(score, 2));
    scores_subset = score(:, 1:n_pcs);
    
    % Calculate Mahalanobis distance
    mahal_dist = mahal(scores_subset, scores_subset);
    
    % Chi-squared threshold
    threshold = chi2inv(confidence, n_pcs);
    
    % Identify outliers
    outlier_mask = mahal_dist > threshold;
end

function mean_corr = calculate_within_sample_correlation(spectra)
    % CALCULATE_WITHIN_SAMPLE_CORRELATION - Calculate mean pairwise correlation
    %
    % Inputs:
    %   spectra - Matrix of spectra [n_spectra × n_wavenumbers]
    %
    % Outputs:
    %   mean_corr - Mean pairwise correlation coefficient
    
    % Calculate correlation matrix
    corr_matrix = corrcoef(spectra');
    
    % Extract upper triangle (excluding diagonal)
    upper_tri = corr_matrix(triu(true(size(corr_matrix)), 1));
    
    % Calculate mean
    mean_corr = mean(upper_tri);
end

function log_message(message, log_file_handle)
    % LOG_MESSAGE - Log a message with timestamp
    %
    % Inputs:
    %   message         - Message to log
    %   log_file_handle - File handle for log file
    
    timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    log_str = sprintf('[%s] %s\n', timestamp, message);
    
    % Write to file
    fprintf(log_file_handle, '%s', log_str);
    
    % Also print to console
    fprintf('%s', log_str);
end