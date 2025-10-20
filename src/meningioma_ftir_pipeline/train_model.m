function [trained_model, training_results, cv_results] = train_model(X_train, y_train, model_params)
    % Train classification model for meningioma FTIR spectra
    % Performs model training and cross-validation
    
    % Initialize results structures
    training_results = struct();
    cv_results = table();
    
    % Set up cross-validation
    cv = cvpartition(y_train, 'KFold', 5);
    
    % Initialize performance metrics
    cv_accuracy = zeros(cv.NumTestSets, 1);
    cv_precision = zeros(cv.NumTestSets, 1);
    cv_recall = zeros(cv.NumTestSets, 1);
    cv_f1 = zeros(cv.NumTestSets, 1);
    
    % Perform cross-validation
    for i = 1:cv.NumTestSets
        % Split data
        train_idx = cv.training(i);
        test_idx = cv.test(i);
        
        % Train model
        fold_model = fit_model(X_train(train_idx, :), y_train(train_idx), model_params);
        
        % Evaluate fold
        [fold_metrics] = evaluate_fold(fold_model, X_train(test_idx, :), y_train(test_idx));
        
        % Store metrics
        cv_accuracy(i) = fold_metrics.accuracy;
        cv_precision(i) = fold_metrics.precision;
        cv_recall(i) = fold_metrics.recall;
        cv_f1(i) = fold_metrics.f1;
    end
    
    % Train final model on full training set
    trained_model = fit_model(X_train, y_train, model_params);
    
    % Store results
    training_results.cv_mean_accuracy = mean(cv_accuracy);
    training_results.cv_std_accuracy = std(cv_accuracy);
    
    % Create CV results table
    cv_results.Fold = (1:cv.NumTestSets)';
    cv_results.Accuracy = cv_accuracy;
    cv_results.Precision = cv_precision;
    cv_results.Recall = cv_recall;
    cv_results.F1Score = cv_f1;
    
    % Save model and results
    save('models/meningioma_ftir_pipeline/trained_model.mat', 'trained_model');
    save('results/meningioma_ftir_pipeline/training_results.mat', 'training_results');
    
    % Save cross-validation results
    writetable(cv_results, 'results/meningioma_ftir_pipeline/cv_performance.csv');
end

function model = fit_model(X, y, params)
    % Fit classification model
    % TODO: Implement model training with specified parameters
    model = fitcsvm(X, y, 'KernelFunction', params.kernel, ...
                   'BoxConstraint', params.C, ...
                   'KernelScale', params.sigma);
end

function metrics = evaluate_fold(model, X, y)
    % Calculate performance metrics for a CV fold
    y_pred = predict(model, X);
    
    metrics = struct();
    metrics.accuracy = sum(y_pred == y) / length(y);
    metrics.precision = calculate_precision(y, y_pred);
    metrics.recall = calculate_recall(y, y_pred);
    metrics.f1 = calculate_f1(metrics.precision, metrics.recall);
end

function precision = calculate_precision(y_true, y_pred)
    % Calculate precision
    true_positives = sum(y_true == 1 & y_pred == 1);
    false_positives = sum(y_true == 0 & y_pred == 1);
    precision = true_positives / (true_positives + false_positives);
end

function recall = calculate_recall(y_true, y_pred)
    % Calculate recall
    true_positives = sum(y_true == 1 & y_pred == 1);
    false_negatives = sum(y_true == 1 & y_pred == 0);
    recall = true_positives / (true_positives + false_negatives);
end

function f1 = calculate_f1(precision, recall)
    % Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall);
end