classdef VisualizationTools < handle
    %VISUALIZATIONTOOLS Create plots and visualizations for CV results
    %
    % DESCRIPTION:
    %   Generates publication-quality visualizations including confusion matrices,
    %   ROC curves, performance heatmaps, and comparison plots.
    %
    % USAGE:
    %   viz = VisualizationTools('OutputDir', 'results/plots');
    %   viz.plot_confusion_matrix(conf_matrix, class_names);
    %   viz.plot_performance_heatmap(summary);
    %
    % See also: ResultsAggregator, ReportGenerator
    
    properties (Access = private)
        output_dir      % Directory for saving plots
        save_plots      % Whether to save plots to disk
        figure_format   % Format for saved figures ('png', 'pdf', 'fig')
    end
    
    methods
        function obj = VisualizationTools(varargin)
            %VISUALIZATIONTOOLS Constructor
            %
            % SYNTAX:
            %   viz = VisualizationTools()
            %   viz = VisualizationTools('OutputDir', 'plots', 'SavePlots', true)
            
            p = inputParser;
            addParameter(p, 'OutputDir', 'plots', @ischar);
            addParameter(p, 'SavePlots', false, @islogical);
            addParameter(p, 'FigureFormat', 'png', @(x) ismember(x, {'png', 'pdf', 'fig', 'eps'}));
            parse(p, varargin{:});
            
            obj.output_dir = p.Results.OutputDir;
            obj.save_plots = p.Results.SavePlots;
            obj.figure_format = p.Results.FigureFormat;
            
            % Create output directory if saving
            if obj.save_plots && ~exist(obj.output_dir, 'dir')
                mkdir(obj.output_dir);
            end
        end
        
        function fig = plot_confusion_matrix(obj, conf_matrix, class_names, varargin)
            %PLOT_CONFUSION_MATRIX Visualize confusion matrix as heatmap
            %
            % SYNTAX:
            %   fig = viz.plot_confusion_matrix(conf_matrix, class_names)
            %   fig = viz.plot_confusion_matrix(conf_matrix, class_names, 'Title', 'My CM')
            
            p = inputParser;
            addRequired(p, 'conf_matrix', @ismatrix);
            addRequired(p, 'class_names', @iscell);
            addParameter(p, 'Title', 'Confusion Matrix', @ischar);
            addParameter(p, 'Normalize', false, @islogical);
            parse(p, conf_matrix, class_names, varargin{:});
            
            cm = conf_matrix;
            if p.Results.Normalize
                cm = cm ./ sum(cm, 2);  % Normalize by row
            end
            
            fig = figure('Position', [100, 100, 600, 500]);
            
            % Create heatmap
            h = heatmap(class_names, class_names, cm);
            h.Title = p.Results.Title;
            h.XLabel = 'Predicted Class';
            h.YLabel = 'True Class';
            h.ColorbarVisible = 'on';
            h.ColorLimits = [0, max(cm(:))];
            
            % Save if requested
            if obj.save_plots
                filename = fullfile(obj.output_dir, ...
                    sprintf('confusion_matrix.%s', obj.figure_format));
                obj.save_figure(fig, filename);
            end
        end
        
        function fig = plot_roc_curve(obj, y_true, scores, varargin)
            %PLOT_ROC_CURVE Plot ROC curves for binary or multi-class classification
            %
            % SYNTAX:
            %   fig = viz.plot_roc_curve(y_true, scores)
            %   fig = viz.plot_roc_curve(y_true, scores, 'Title', 'ROC Analysis')
            
            p = inputParser;
            addRequired(p, 'y_true');
            addRequired(p, 'scores', @ismatrix);
            addParameter(p, 'Title', 'ROC Curve', @ischar);
            parse(p, y_true, scores, varargin{:});
            
            fig = figure('Position', [100, 100, 600, 500]);
            hold on;
            
            classes = categories(y_true);
            n_classes = length(classes);
            colors = lines(n_classes);
            
            if n_classes == 2
                % Binary classification
                [fpr, tpr, ~, auc] = perfcurve(y_true, scores(:, 2), classes{2});
                plot(fpr, tpr, 'LineWidth', 2, 'Color', colors(1, :), ...
                    'DisplayName', sprintf('AUC = %.3f', auc));
            else
                % Multi-class: one-vs-rest
                for c = 1:n_classes
                    y_binary = y_true == categorical(cellstr(classes{c}));
                    if sum(y_binary) > 0 && sum(~y_binary) > 0
                        try
                            [fpr, tpr, ~, auc] = perfcurve(y_binary, scores(:, c), true);
                            plot(fpr, tpr, 'LineWidth', 2, 'Color', colors(c, :), ...
                                'DisplayName', sprintf('%s (AUC=%.3f)', classes{c}, auc));
                        catch
                            % Skip if perfcurve fails
                        end
                    end
                end
            end
            
            % Diagonal reference line
            plot([0, 1], [0, 1], 'k--', 'LineWidth', 1, 'DisplayName', 'Random');
            
            xlabel('False Positive Rate');
            ylabel('True Positive Rate');
            title(p.Results.Title);
            legend('Location', 'southeast');
            grid on;
            axis square;
            hold off;
            
            % Save if requested
            if obj.save_plots
                filename = fullfile(obj.output_dir, ...
                    sprintf('roc_curve.%s', obj.figure_format));
                obj.save_figure(fig, filename);
            end
        end
        
        function fig = plot_performance_heatmap(obj, summary, metric_name)
            %PLOT_PERFORMANCE_HEATMAP Heatmap of metric across configurations
            %
            % SYNTAX:
            %   fig = viz.plot_performance_heatmap(summary, 'accuracy')
            
            % Extract data
            n_perms = size(summary.configurations, 1);
            n_clfs = size(summary.configurations, 2);
            
            performance_matrix = zeros(n_perms, n_clfs);
            perm_labels = cell(n_perms, 1);
            clf_labels = cell(n_clfs, 1);
            
            for p = 1:n_perms
                for c = 1:n_clfs
                    config = summary.configurations{p, c};
                    performance_matrix(p, c) = config.mean_metrics.(metric_name);
                    
                    if c == 1
                        perm_labels{p} = config.permutation_id;
                    end
                    if p == 1
                        clf_labels{c} = config.classifier_name;
                    end
                end
            end
            
            fig = figure('Position', [100, 100, 800, 600]);
            
            h = heatmap(clf_labels, perm_labels, performance_matrix);
            h.Title = sprintf('%s Performance Across Configurations', strrep(metric_name, '_', ' '));
            h.XLabel = 'Classifier';
            h.YLabel = 'Preprocessing Permutation';
            h.ColorbarVisible = 'on';
            
            % Save if requested
            if obj.save_plots
                filename = fullfile(obj.output_dir, ...
                    sprintf('performance_heatmap_%s.%s', metric_name, obj.figure_format));
                obj.save_figure(fig, filename);
            end
        end
        
        function fig = plot_classifier_comparison(obj, comparison)
            %PLOT_CLASSIFIER_COMPARISON Bar plot comparing classifiers
            %
            % SYNTAX:
            %   fig = viz.plot_classifier_comparison(comparison_struct)
            
            n_clfs = length(comparison.classifiers);
            mean_scores = mean(comparison.mean_scores, 2);
            std_scores = std(comparison.mean_scores, 0, 2);
            
            fig = figure('Position', [100, 100, 700, 500]);
            
            b = bar(1:n_clfs, mean_scores);
            hold on;
            errorbar(1:n_clfs, mean_scores, std_scores, 'k.', 'LineWidth', 1.5);
            hold off;
            
            set(gca, 'XTick', 1:n_clfs, 'XTickLabel', comparison.classifiers);
            xlabel('Classifier');
            ylabel(sprintf('%s', strrep(comparison.metric, '_', ' ')));
            title(sprintf('Classifier Comparison (%s-level)', comparison.level));
            grid on;
            
            % Save if requested
            if obj.save_plots
                filename = fullfile(obj.output_dir, ...
                    sprintf('classifier_comparison_%s.%s', comparison.metric, obj.figure_format));
                obj.save_figure(fig, filename);
            end
        end
        
        function fig = plot_permutation_comparison(obj, summary, metric_name)
            %PLOT_PERMUTATION_COMPARISON Compare preprocessing permutations
            %
            % SYNTAX:
            %   fig = viz.plot_permutation_comparison(summary, 'accuracy')
            
            n_perms = size(summary.configurations, 1);
            n_clfs = size(summary.configurations, 2);
            
            % Collect data
            perm_labels = cell(n_perms, 1);
            perm_means = zeros(n_perms, n_clfs);
            
            for p = 1:n_perms
                for c = 1:n_clfs
                    config = summary.configurations{p, c};
                    if p <= length(perm_labels) && isempty(perm_labels{p})
                        perm_labels{p} = config.permutation_id;
                    end
                    perm_means(p, c) = config.mean_metrics.(metric_name);
                end
            end
            
            fig = figure('Position', [100, 100, 800, 500]);
            
            % Grouped bar plot
            b = bar(perm_means);
            
            % Get classifier names for legend
            clf_names = cell(n_clfs, 1);
            for c = 1:n_clfs
                clf_names{c} = summary.configurations{1, c}.classifier_name;
            end
            
            set(gca, 'XTick', 1:n_perms, 'XTickLabel', perm_labels);
            xlabel('Preprocessing Permutation');
            ylabel(strrep(metric_name, '_', ' '));
            title(sprintf('Permutation Comparison: %s', strrep(metric_name, '_', ' ')));
            legend(clf_names, 'Location', 'best');
            grid on;
            
            % Save if requested
            if obj.save_plots
                filename = fullfile(obj.output_dir, ...
                    sprintf('permutation_comparison_%s.%s', metric_name, obj.figure_format));
                obj.save_figure(fig, filename);
            end
        end
        
        function fig = plot_metric_boxplots(obj, summary, metric_name)
            %PLOT_METRIC_BOXPLOTS Boxplots showing metric distribution across folds
            %
            % SYNTAX:
            %   fig = viz.plot_metric_boxplots(summary, 'accuracy')
            
            n_configs = numel(summary.configurations);
            
            % Collect all fold values
            all_data = [];
            group_labels = {};
            
            for i = 1:n_configs
                config = summary.configurations{i};
                
                % Extract metric from all folds
                fold_values = zeros(length(config.all_fold_metrics), 1);
                for j = 1:length(config.all_fold_metrics)
                    if isfield(config.all_fold_metrics{j}, metric_name)
                        fold_values(j) = config.all_fold_metrics{j}.(metric_name);
                    else
                        fold_values(j) = NaN;
                    end
                end
                
                all_data = [all_data; fold_values]; %#ok<AGROW>
                
                % Create label
                label = sprintf('%s+%s', config.permutation_id, config.classifier_name);
                group_labels = [group_labels; repmat({label}, length(fold_values), 1)]; %#ok<AGROW>
            end
            
            fig = figure('Position', [100, 100, 1000, 600]);
            boxplot(all_data, group_labels, 'LabelOrientation', 'inline');
            ylabel(strrep(metric_name, '_', ' '));
            title(sprintf('%s Distribution Across Folds', strrep(metric_name, '_', ' ')));
            grid on;
            
            % Rotate x-labels if many configurations
            if n_configs > 4
                set(gca, 'XTickLabelRotation', 45);
            end
            
            % Save if requested
            if obj.save_plots
                filename = fullfile(obj.output_dir, ...
                    sprintf('boxplot_%s.%s', metric_name, obj.figure_format));
                obj.save_figure(fig, filename);
            end
        end
    end
    
    methods (Access = private)
        function save_figure(obj, fig, filename)
            %SAVE_FIGURE Save figure to file
            
            try
                switch obj.figure_format
                    case 'png'
                        print(fig, filename, '-dpng', '-r300');
                    case 'pdf'
                        print(fig, filename, '-dpdf', '-r300');
                    case 'eps'
                        print(fig, filename, '-depsc', '-r300');
                    case 'fig'
                        savefig(fig, filename);
                end
                fprintf('Saved: %s\n', filename);
            catch ME
                warning('Failed to save figure: %s', ME.message);
            end
        end
    end
end
