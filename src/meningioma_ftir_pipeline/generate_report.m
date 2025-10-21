%% PHASE 6: REPORT GENERATION
% This script generates a comprehensive report of the model development and evaluation.

function generate_report(cfg, cv_results, final_model, test_results)
    fprintf('Generating report summary...\n');
    
    % Load data if not provided
    if nargin < 2
        fprintf('Loading results data...\n');
        load(fullfile(cfg.paths.results, 'best_classifier_selection.mat'), 'best_model_info');
        load(fullfile(cfg.paths.models, 'final_model.mat'), 'final_model_package');
        load(fullfile(cfg.paths.results, 'test_results.mat'), 'test_results');
    else
        best_model_info = [];  % Will be extracted from cv_results
        final_model_package = final_model;
    end
    
    % Create a text-only summary report
    report_file = fullfile(cfg.paths.results, 'model_report_summary.txt');
    fid = fopen(report_file, 'w');
    
    if fid == -1
        error('Could not open report file for writing: %s', report_file);
    end
    
    % Print header
    fprintf(fid, '=======================================================\n');
    fprintf(fid, '      MENINGIOMA FT-IR CLASSIFICATION MODEL REPORT     \n');
    fprintf(fid, '=======================================================\n\n');
    fprintf(fid, 'Report generated on: %s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    
    % Check if we have a model directly or a package structure
    if isa(final_model, 'struct') && isfield(final_model, 'model')
        % We have a package structure with a model field
        model = final_model.model;
        classifier_type = final_model.classifier_type;
        if isfield(final_model, 'n_training_samples')
            n_training_samples = final_model.n_training_samples;
        else
            n_training_samples = 'Unknown';
        end
    else
        % The model was passed directly
        model = final_model;
        % Try to determine the classifier type from the model
        if isa(model, 'TreeBagger')
            classifier_type = 'RandomForest';
            n_training_samples = size(model.X, 1);
        elseif isa(model, 'ClassificationDiscriminant')
            classifier_type = 'LDA';
            n_training_samples = size(model.X, 1);
        elseif isa(model, 'ClassificationSVM')
            classifier_type = 'SVM';
            n_training_samples = size(model.X, 1);
        elseif isa(model, 'ClassificationPLS')
            classifier_type = 'PLSDA';
            n_training_samples = size(model.X, 1);
        else
            classifier_type = 'Unknown';
            n_training_samples = 'Unknown';
        end
    end

    % 1. Data summary
    fprintf(fid, '1. DATA SUMMARY\n');
    fprintf(fid, '----------------\n');
    fprintf(fid, 'Training samples: %s\n', num2str(n_training_samples));
    fprintf(fid, 'Test samples: %d\n', length(test_results.predictions));
    fprintf(fid, '\n');
    
    % 2. Model Selection
    fprintf(fid, '2. MODEL SELECTION\n');
    fprintf(fid, '------------------\n');
    
    % Extract best model info from cv_results if provided
    if ~isempty(cv_results)
        classifier_names = {'LDA', 'PLSDA', 'SVM', 'RandomForest'};
        mean_f2_scores = zeros(length(cv_results), 1);
        
        fprintf(fid, 'Cross-validation results:\n');
        for i = 1:length(cv_results)
            if isfield(cv_results{i}, 'performance') && ~isempty(cv_results{i}.performance)
                % Extract F2 scores
                all_f2 = [];
                for j = 1:length(cv_results{i}.performance)
                    if ~isempty(cv_results{i}.performance{j}) && isfield(cv_results{i}.performance{j}, 'f2')
                        all_f2 = [all_f2, cv_results{i}.performance{j}.f2];
                    end
                end
                mean_f2 = mean(all_f2);
                std_f2 = std(all_f2);
                mean_f2_scores(i) = mean_f2;
                
                fprintf(fid, '  %s: F2 = %.3f ± %.3f\n', ...
                        classifier_names{i}, mean_f2, std_f2);
            end
        end
        
        % Find best classifier
        [~, best_idx] = max(mean_f2_scores);
        best_classifier = classifier_names{best_idx};
        
        fprintf(fid, '\nSelected model: %s (F2 score: %.3f)\n', ...
                best_classifier, mean_f2_scores(best_idx));
    else
        fprintf(fid, 'Selected model: %s\n', final_model_package.classifier_type);
    end
    
    % Print hyperparameters
    fprintf(fid, '\nModel hyperparameters:\n');
    if isa(final_model, 'struct') && isfield(final_model, 'hyperparameters')
        params = final_model.hyperparameters;
        param_fields = fieldnames(params);
        for i = 1:length(param_fields)
            fprintf(fid, '  %s = %s\n', param_fields{i}, mat2str(params.(param_fields{i})));
        end
    elseif isa(model, 'TreeBagger')
        fprintf(fid, '  n_trees = %d\n', model.NumTrees);
        fprintf(fid, '  max_depth = %s\n', 'Unknown (in model object)');
    elseif isa(model, 'ClassificationSVM')
        fprintf(fid, '  kernel = %s\n', model.KernelFunction);
        fprintf(fid, '  C = %f\n', model.BoxConstraints(1));
    else
        fprintf(fid, '  (No hyperparameters available)\n');
    end
    fprintf(fid, '\n');
    
    % 3. Test Set Performance
    fprintf(fid, '3. TEST SET PERFORMANCE\n');
    fprintf(fid, '-----------------------\n');
    
    % Confusion matrix
    fprintf(fid, 'Confusion Matrix:\n');
    fprintf(fid, '            Predicted WHO-1  Predicted WHO-3\n');
    fprintf(fid, 'True WHO-1      %6d           %6d\n', ...
            test_results.confusion_matrix(1,1), test_results.confusion_matrix(1,2));
    fprintf(fid, 'True WHO-3      %6d           %6d\n', ...
            test_results.confusion_matrix(2,1), test_results.confusion_matrix(2,2));
    fprintf(fid, '\n');
    
    % Performance metrics
    fprintf(fid, 'Performance Metrics:\n');
    fprintf(fid, '  Accuracy: %.2f%%\n', test_results.accuracy * 100);
    fprintf(fid, '  Balanced Accuracy: %.2f%%\n', test_results.balanced_accuracy * 100);
    fprintf(fid, '  Sensitivity (WHO-3): %.2f%%\n', test_results.sensitivity * 100);
    fprintf(fid, '  Specificity (WHO-1): %.2f%%\n', test_results.specificity * 100);
    fprintf(fid, '  Precision (PPV): %.2f%%\n', test_results.ppv * 100);
    fprintf(fid, '  F1 Score: %.3f\n', test_results.f1);
    fprintf(fid, '  F2 Score: %.3f\n', test_results.f2);
    fprintf(fid, '  AUC-ROC: %.3f\n', test_results.auc);
    fprintf(fid, '\n');
    
    % 4. Conclusion
    fprintf(fid, '4. CONCLUSION\n');
    fprintf(fid, '-------------\n');
    
    % Interpret the results
    if test_results.f2 > 0.7
        performance = 'excellent';
    elseif test_results.f2 > 0.6
        performance = 'good';
    elseif test_results.f2 > 0.5
        performance = 'fair';
    else
        performance = 'poor';
    end
    
    fprintf(fid, 'The %s model shows %s performance for\n', ...
            classifier_type, performance);
    fprintf(fid, 'classifying meningioma samples based on FT-IR spectroscopy data.\n');
    
    if test_results.sensitivity < 0.7
        fprintf(fid, '\nThe sensitivity for WHO-3 tumors is relatively low (%.2f). \n', test_results.sensitivity);
        fprintf(fid, 'Consider collecting more high-grade samples for model improvement.\n');
    end
    
    if test_results.specificity < 0.7
        fprintf(fid, '\nThe specificity for WHO-1 tumors is relatively low (%.2f). \n', test_results.specificity);
        fprintf(fid, 'Consider additional feature engineering to improve differentiation.\n');
    end
    
    % Close the report file
    fclose(fid);
    
    fprintf('Report generated successfully: %s\n', report_file);
end