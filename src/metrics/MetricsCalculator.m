classdef MetricsCalculator < handle
    %METRICSCALCULATOR Compute spectrum-level and patient-level classification metrics
    %
    % DESCRIPTION:
    %   Calculates comprehensive performance metrics at both sample and
    %   patient levels. Patient-level metrics use majority voting aggregation.
    %
    % USAGE:
    %   calc = MetricsCalculator('Verbose', true);
    %   metrics = calc.compute_spectrum_metrics(y_true, y_pred, scores);
    %   patient_metrics = calc.compute_patient_metrics(y_true, y_pred, scores, patient_ids);
    %
    % METRICS COMPUTED:
    %   Spectrum-level: accuracy, sensitivity, specificity, precision, F1, AUC-ROC
    %   Patient-level: Same metrics after majority vote aggregation
    %
    % See also: CrossValidationEngine, ClassifierWrapper
    
    properties (Access = private)
        verbose  % Display progress
    end
    
    methods
        function obj = MetricsCalculator(varargin)
            %METRICSCALCULATOR Constructor
            %
            % SYNTAX:
            %   calc = MetricsCalculator()
            %   calc = MetricsCalculator('Verbose', true)
            
            p = inputParser;
            addParameter(p, 'Verbose', false, @islogical);
            parse(p, varargin{:});
            
            obj.verbose = p.Results.Verbose;
        end
        
        function metrics = compute_spectrum_metrics(obj, y_true, y_pred, scores)
            %COMPUTE_SPECTRUM_METRICS Calculate sample-level classification metrics
            %
            % SYNTAX:
            %   metrics = calc.compute_spectrum_metrics(y_true, y_pred, scores)
            %
            % INPUTS:
            %   y_true: True labels (categorical)
            %   y_pred: Predicted labels (categorical)
            %   scores: Prediction scores/probabilities [n_samples × n_classes]
            %
            % OUTPUTS:
            %   metrics: Structure with accuracy, confusion matrix, per-class metrics, AUC
            
            metrics = struct();
            
            % Convert categorical to numeric for processing
            classes = categories(y_true);
            n_classes = length(classes);
            
            % Overall accuracy
            metrics.accuracy = sum(y_true == y_pred) / length(y_true);
            
            % Confusion matrix
            metrics.confusion_matrix = confusionmat(y_true, y_pred);
            
            % Per-class metrics
            metrics.per_class = struct();
            for c = 1:n_classes
                class_name = classes{c};
                
                % Create valid field name
                field_name = matlab.lang.makeValidName(['class_' class_name]);
                
                % True positives, false positives, etc.
                tp = sum((y_true == categorical(cellstr(class_name))) & (y_pred == categorical(cellstr(class_name))));
                fp = sum((y_true ~= categorical(cellstr(class_name))) & (y_pred == categorical(cellstr(class_name))));
                tn = sum((y_true ~= categorical(cellstr(class_name))) & (y_pred ~= categorical(cellstr(class_name))));
                fn = sum((y_true == categorical(cellstr(class_name))) & (y_pred ~= categorical(cellstr(class_name))));
                
                % Sensitivity (recall)
                if (tp + fn) > 0
                    sensitivity = tp / (tp + fn);
                else
                    sensitivity = NaN;
                end
                
                % Specificity
                if (tn + fp) > 0
                    specificity = tn / (tn + fp);
                else
                    specificity = NaN;
                end
                
                % Precision
                if (tp + fp) > 0
                    precision = tp / (tp + fp);
                else
                    precision = NaN;
                end
                
                % F1 score
                if (precision + sensitivity) > 0
                    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity);
                else
                    f1_score = NaN;
                end
                
                % Store metrics
                metrics.per_class.(field_name) = struct(...
                    'sensitivity', sensitivity, ...
                    'specificity', specificity, ...
                    'precision', precision, ...
                    'f1_score', f1_score, ...
                    'tp', tp, 'fp', fp, 'tn', tn, 'fn', fn);
            end
            
            % Macro-averaged metrics
            all_sens = zeros(n_classes, 1);
            all_spec = zeros(n_classes, 1);
            all_prec = zeros(n_classes, 1);
            all_f1 = zeros(n_classes, 1);
            
            for c = 1:n_classes
                class_name = classes{c};
                field_name = matlab.lang.makeValidName(['class_' class_name]);
                class_metrics = metrics.per_class.(field_name);
                all_sens(c) = class_metrics.sensitivity;
                all_spec(c) = class_metrics.specificity;
                all_prec(c) = class_metrics.precision;
                all_f1(c) = class_metrics.f1_score;
            end
            
            metrics.macro_sensitivity = mean(all_sens, 'omitnan');
            metrics.macro_specificity = mean(all_spec, 'omitnan');
            metrics.macro_precision = mean(all_prec, 'omitnan');
            metrics.macro_f1 = mean(all_f1, 'omitnan');
            
            % AUC-ROC (multi-class: one-vs-rest)
            if n_classes == 2
                % Binary classification
                % Assuming scores for positive class are in second column
                if size(scores, 2) == 2
                    [~, ~, ~, auc] = perfcurve(y_true, scores(:, 2), classes{2});
                    metrics.auc = auc;
                else
                    metrics.auc = NaN;
                end
            else
                % Multi-class: compute AUC for each class vs rest
                auc_per_class = zeros(n_classes, 1);
                for c = 1:n_classes
                    % Create binary labels (this class vs others)
                    class_label = classes{c};
                    y_binary = y_true == categorical(cellstr(class_label));
                    if sum(y_binary) > 0 && sum(~y_binary) > 0
                        try
                            [~, ~, ~, auc] = perfcurve(y_binary, scores(:, c), true);
                            auc_per_class(c) = auc;
                        catch
                            auc_per_class(c) = NaN;
                        end
                    else
                        auc_per_class(c) = NaN;
                    end
                end
                metrics.auc = mean(auc_per_class, 'omitnan');  % Macro-average
                metrics.auc_per_class = auc_per_class;
            end
            
            % Store metadata
            metrics.n_samples = length(y_true);
            metrics.n_classes = n_classes;
            metrics.classes = classes;
        end
        
        function patient_metrics = compute_patient_metrics(obj, y_true, y_pred, scores, patient_ids)
            %COMPUTE_PATIENT_METRICS Calculate patient-level metrics using majority voting
            %
            % SYNTAX:
            %   patient_metrics = calc.compute_patient_metrics(y_true, y_pred, scores, patient_ids)
            %
            % INPUTS:
            %   y_true: True labels (categorical) [n_samples × 1]
            %   y_pred: Predicted labels (categorical) [n_samples × 1]
            %   scores: Prediction scores [n_samples × n_classes]
            %   patient_ids: Patient identifiers [n_samples × 1]
            %
            % OUTPUTS:
            %   patient_metrics: Structure with aggregated patient-level metrics
            
            % Get unique patients
            unique_patients = unique(patient_ids);
            n_patients = length(unique_patients);
            
            % Initialize patient-level predictions
            patient_y_true = categorical(zeros(n_patients, 1));
            patient_y_pred = categorical(zeros(n_patients, 1));
            patient_scores = zeros(n_patients, size(scores, 2));
            
            % Aggregate per patient
            for i = 1:n_patients
                pid = unique_patients(i);
                patient_mask = patient_ids == pid;
                
                % True label for this patient (should all be same)
                patient_y_true(i) = mode(y_true(patient_mask));
                
                % Predicted label: majority vote
                patient_y_pred(i) = mode(y_pred(patient_mask));
                
                % Average scores across samples
                patient_scores(i, :) = mean(scores(patient_mask, :), 1);
            end
            
            % Compute metrics on aggregated patient-level data
            patient_metrics = obj.compute_spectrum_metrics(patient_y_true, ...
                patient_y_pred, patient_scores);
            
            % Add patient-specific metadata
            patient_metrics.n_patients = n_patients;
            patient_metrics.aggregation_method = 'majority_vote';
        end
        
        function display_metrics(obj, metrics, level_name)
            %DISPLAY_METRICS Pretty-print metrics
            %
            % SYNTAX:
            %   calc.display_metrics(metrics, 'Spectrum-level')
            
            fprintf('\n=== %s METRICS ===\n', upper(level_name));
            fprintf('Accuracy: %.4f\n', metrics.accuracy);
            fprintf('Macro F1: %.4f\n', metrics.macro_f1);
            fprintf('Macro Sensitivity: %.4f\n', metrics.macro_sensitivity);
            fprintf('Macro Specificity: %.4f\n', metrics.macro_specificity);
            fprintf('AUC: %.4f\n', metrics.auc);
            
            fprintf('\nConfusion Matrix:\n');
            disp(metrics.confusion_matrix);
            
            fprintf('\nPer-class metrics:\n');
            for c = 1:metrics.n_classes
                class_name = metrics.classes{c};
                field_name = matlab.lang.makeValidName(['class_' class_name]);
                class_metrics = metrics.per_class.(field_name);
                fprintf('  Class %s: Sens=%.4f, Spec=%.4f, Prec=%.4f, F1=%.4f\n', ...
                    class_name, class_metrics.sensitivity, class_metrics.specificity, ...
                    class_metrics.precision, class_metrics.f1_score);
            end
        end
    end
end
