classdef ResultsAggregator < handle
    %RESULTSAGGREGATOR Aggregate and summarize cross-validation results
    %
    % DESCRIPTION:
    %   Aggregates metrics across folds, repeats, classifiers, and preprocessing
    %   permutations. Computes summary statistics and prepares data for
    %   visualization and statistical comparison.
    %
    % USAGE:
    %   aggregator = ResultsAggregator(cv_results);
    %   summary = aggregator.summarize();
    %   best = aggregator.get_best_configuration('accuracy');
    %
    % See also: CrossValidationEngine, MetricsCalculator, ReportGenerator
    
    properties (Access = private)
        cv_results      % Cross-validation results structure
        metrics_calc    % MetricsCalculator instance
        verbose         % Display progress
    end
    
    methods
        function obj = ResultsAggregator(cv_results, varargin)
            %RESULTSAGGREGATOR Constructor
            %
            % SYNTAX:
            %   aggregator = ResultsAggregator(cv_results)
            %   aggregator = ResultsAggregator(cv_results, 'Verbose', true)
            
            p = inputParser;
            addRequired(p, 'cv_results', @isstruct);
            addParameter(p, 'Verbose', false, @islogical);
            parse(p, cv_results, varargin{:});
            
            obj.cv_results = cv_results;
            obj.verbose = p.Results.Verbose;
            obj.metrics_calc = MetricsCalculator('Verbose', false);
        end
        
        function summary = summarize(obj, varargin)
            %SUMMARIZE Compute summary statistics across all CV folds
            %
            % SYNTAX:
            %   summary = aggregator.summarize()
            %   summary = aggregator.summarize('Level', 'patient')
            %
            % INPUTS:
            %   Level: 'spectrum' or 'patient' (default: 'spectrum')
            %
            % OUTPUTS:
            %   summary: Structure with mean/std/median metrics per configuration
            
            p = inputParser;
            addParameter(p, 'Level', 'spectrum', @(x) ismember(x, {'spectrum', 'patient'}));
            parse(p, varargin{:});
            
            level = p.Results.Level;
            
            if obj.verbose
                fprintf('\n=== AGGREGATING RESULTS (%s-level) ===\n', upper(level));
            end
            
            summary = struct();
            summary.level = level;
            summary.n_permutations = obj.cv_results.n_permutations;
            summary.n_classifiers = obj.cv_results.n_classifiers;
            summary.n_folds = obj.cv_results.n_folds;
            summary.n_repeats = obj.cv_results.n_repeats;
            summary.configurations = cell(obj.cv_results.n_permutations, obj.cv_results.n_classifiers);
            
            % Aggregate for each permutation-classifier combination
            for p = 1:obj.cv_results.n_permutations
                perm = obj.cv_results.permutations{p};
                perm_id = perm.permutation_id;
                
                for c = 1:length(perm.classifiers)
                    clf = perm.classifiers{c};
                    clf_name = clf.classifier_name;
                    
                    if obj.verbose
                        fprintf('Processing: %s + %s\n', perm_id, clf_name);
                    end
                    
                    % Aggregate metrics across all folds and repeats
                    config_summary = obj.aggregate_configuration(clf, level);
                    config_summary.permutation_id = perm_id;
                    config_summary.classifier_name = clf_name;
                    
                    summary.configurations{p, c} = config_summary;
                end
            end
            
            if obj.verbose
                fprintf('Aggregation complete\n');
            end
        end
        
        function best = get_best_configuration(obj, metric_name, varargin)
            %GET_BEST_CONFIGURATION Find best performing configuration
            %
            % SYNTAX:
            %   best = aggregator.get_best_configuration('accuracy')
            %   best = aggregator.get_best_configuration('macro_f1', 'Level', 'patient')
            %
            % INPUTS:
            %   metric_name: Metric to optimize ('accuracy', 'macro_f1', 'auc', etc.)
            %   Level: 'spectrum' or 'patient'
            %
            % OUTPUTS:
            %   best: Structure with best configuration details
            
            p = inputParser;
            addRequired(p, 'metric_name', @ischar);
            addParameter(p, 'Level', 'spectrum', @(x) ismember(x, {'spectrum', 'patient'}));
            parse(p, metric_name, varargin{:});
            
            level = p.Results.Level;
            
            % Get summary
            summary = obj.summarize('Level', level);
            
            % Find best configuration
            best_value = -inf;
            best_config = struct();
            
            for p = 1:size(summary.configurations, 1)
                for c = 1:size(summary.configurations, 2)
                    config = summary.configurations{p, c};
                    
                    if isfield(config.mean_metrics, metric_name)
                        value = config.mean_metrics.(metric_name);
                        if value > best_value
                            best_value = value;
                            best_config = config;
                        end
                    end
                end
            end
            
            best = best_config;
            best.optimized_metric = metric_name;
            best.best_value = best_value;
            
            if obj.verbose
                fprintf('\nBest configuration for %s:\n', metric_name);
                fprintf('  Permutation: %s\n', best.permutation_id);
                fprintf('  Classifier: %s\n', best.classifier_name);
                fprintf('  %s = %.4f (± %.4f)\n', metric_name, best_value, ...
                    best.std_metrics.(metric_name));
            end
        end
        
        function comparison = compare_classifiers(obj, varargin)
            %COMPARE_CLASSIFIERS Statistical comparison of classifiers
            %
            % SYNTAX:
            %   comparison = aggregator.compare_classifiers()
            %   comparison = aggregator.compare_classifiers('Metric', 'accuracy')
            %
            % OUTPUTS:
            %   comparison: Structure with pairwise comparison results
            
            p = inputParser;
            addParameter(p, 'Metric', 'accuracy', @ischar);
            addParameter(p, 'Level', 'spectrum', @(x) ismember(x, {'spectrum', 'patient'}));
            parse(p, varargin{:});
            
            metric_name = p.Results.Metric;
            level = p.Results.Level;
            
            summary = obj.summarize('Level', level);
            
            comparison = struct();
            comparison.metric = metric_name;
            comparison.level = level;
            comparison.classifiers = cell(obj.cv_results.n_classifiers, 1);
            comparison.mean_scores = zeros(obj.cv_results.n_classifiers, obj.cv_results.n_permutations);
            
            % Extract mean scores per classifier across permutations
            for c = 1:obj.cv_results.n_classifiers
                clf_name = summary.configurations{1, c}.classifier_name;
                comparison.classifiers{c} = clf_name;
                
                for p = 1:obj.cv_results.n_permutations
                    config = summary.configurations{p, c};
                    if isfield(config.mean_metrics, metric_name)
                        comparison.mean_scores(c, p) = config.mean_metrics.(metric_name);
                    else
                        comparison.mean_scores(c, p) = NaN;
                    end
                end
            end
            
            if obj.verbose
                fprintf('\nClassifier comparison (%s):\n', metric_name);
                for c = 1:length(comparison.classifiers)
                    fprintf('  %s: %.4f (± %.4f)\n', comparison.classifiers{c}, ...
                        mean(comparison.mean_scores(c, :), 'omitnan'), ...
                        std(comparison.mean_scores(c, :), 'omitnan'));
                end
            end
        end
        
        function table_data = to_table(obj, varargin)
            %TO_TABLE Convert summary to MATLAB table for export
            %
            % SYNTAX:
            %   tbl = aggregator.to_table()
            %   tbl = aggregator.to_table('Level', 'patient')
            
            p = inputParser;
            addParameter(p, 'Level', 'spectrum', @(x) ismember(x, {'spectrum', 'patient'}));
            parse(p, varargin{:});
            
            level = p.Results.Level;
            summary = obj.summarize('Level', level);
            
            % Initialize table columns
            permutation_ids = {};
            classifier_names = {};
            mean_acc = [];
            std_acc = [];
            mean_f1 = [];
            std_f1 = [];
            mean_auc = [];
            std_auc = [];
            
            % Populate table
            for p = 1:size(summary.configurations, 1)
                for c = 1:size(summary.configurations, 2)
                    config = summary.configurations{p, c};
                    
                    permutation_ids{end+1} = config.permutation_id; %#ok<AGROW>
                    classifier_names{end+1} = config.classifier_name; %#ok<AGROW>
                    mean_acc(end+1) = config.mean_metrics.accuracy; %#ok<AGROW>
                    std_acc(end+1) = config.std_metrics.accuracy; %#ok<AGROW>
                    mean_f1(end+1) = config.mean_metrics.macro_f1; %#ok<AGROW>
                    std_f1(end+1) = config.std_metrics.macro_f1; %#ok<AGROW>
                    mean_auc(end+1) = config.mean_metrics.auc; %#ok<AGROW>
                    std_auc(end+1) = config.std_metrics.auc; %#ok<AGROW>
                end
            end
            
            % Create table
            table_data = table(permutation_ids', classifier_names', ...
                mean_acc', std_acc', mean_f1', std_f1', mean_auc', std_auc', ...
                'VariableNames', {'Permutation', 'Classifier', ...
                'MeanAccuracy', 'StdAccuracy', 'MeanF1', 'StdF1', 'MeanAUC', 'StdAUC'});
        end
    end
    
    methods (Access = private)
        function config_summary = aggregate_configuration(obj, clf_results, level)
            %AGGREGATE_CONFIGURATION Aggregate metrics for one configuration
            
            n_repeats = length(clf_results.repeats);
            n_folds = length(clf_results.repeats{1}.folds);
            
            % Collect all fold metrics
            all_metrics = cell(n_repeats * n_folds, 1);
            idx = 1;
            
            for r = 1:n_repeats
                repeat = clf_results.repeats{r};
                for f = 1:n_folds
                    fold = repeat.folds{f};
                    
                    % Compute metrics for this fold
                    if strcmp(level, 'spectrum')
                        metrics = obj.metrics_calc.compute_spectrum_metrics(...
                            fold.y_true, fold.y_pred, fold.scores);
                    else
                        metrics = obj.metrics_calc.compute_patient_metrics(...
                            fold.y_true, fold.y_pred, fold.scores, fold.patient_ids);
                    end
                    
                    all_metrics{idx} = metrics;
                    idx = idx + 1;
                end
            end
            
            % Aggregate across folds
            config_summary = struct();
            config_summary.n_folds_total = n_repeats * n_folds;
            
            % Extract metric values
            metric_names = {'accuracy', 'macro_f1', 'macro_sensitivity', ...
                'macro_specificity', 'macro_precision', 'auc'};
            
            config_summary.mean_metrics = struct();
            config_summary.std_metrics = struct();
            config_summary.median_metrics = struct();
            
            for i = 1:length(metric_names)
                metric_name = metric_names{i};
                values = zeros(length(all_metrics), 1);
                
                for j = 1:length(all_metrics)
                    if isfield(all_metrics{j}, metric_name)
                        values(j) = all_metrics{j}.(metric_name);
                    else
                        values(j) = NaN;
                    end
                end
                
                config_summary.mean_metrics.(metric_name) = mean(values, 'omitnan');
                config_summary.std_metrics.(metric_name) = std(values, 'omitnan');
                config_summary.median_metrics.(metric_name) = median(values, 'omitnan');
            end
            
            % Store all fold metrics for further analysis
            config_summary.all_fold_metrics = all_metrics;
        end
    end
end
