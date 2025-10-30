classdef ReportGenerator < handle
    %REPORTGENERATOR Generate comprehensive analysis reports
    %
    % DESCRIPTION:
    %   Orchestrates results aggregation, visualization, and report generation.
    %   Creates both visual and tabular outputs for publication.
    %
    % USAGE:
    %   reporter = ReportGenerator(cv_results, 'OutputDir', 'results/report');
    %   reporter.generate_full_report();
    %
    % See also: ResultsAggregator, VisualizationTools, MetricsCalculator
    
    properties (Access = private)
        cv_results      % Cross-validation results
        aggregator      % ResultsAggregator instance
        visualizer      % VisualizationTools instance
        output_dir      % Output directory
        verbose         % Display progress
    end
    
    methods
        function obj = ReportGenerator(cv_results, varargin)
            %REPORTGENERATOR Constructor
            %
            % SYNTAX:
            %   reporter = ReportGenerator(cv_results)
            %   reporter = ReportGenerator(cv_results, 'OutputDir', 'report', 'Verbose', true)
            
            p = inputParser;
            addRequired(p, 'cv_results', @isstruct);
            addParameter(p, 'OutputDir', 'report', @ischar);
            addParameter(p, 'Verbose', true, @islogical);
            addParameter(p, 'SavePlots', true, @islogical);
            parse(p, cv_results, varargin{:});
            
            obj.cv_results = cv_results;
            obj.output_dir = p.Results.OutputDir;
            obj.verbose = p.Results.Verbose;
            
            % Create output directory
            if ~exist(obj.output_dir, 'dir')
                mkdir(obj.output_dir);
            end
            
            % Initialize components
            obj.aggregator = ResultsAggregator(cv_results, 'Verbose', obj.verbose);
            
            plots_dir = fullfile(obj.output_dir, 'plots');
            obj.visualizer = VisualizationTools('OutputDir', plots_dir, ...
                'SavePlots', p.Results.SavePlots, 'FigureFormat', 'png');
        end
        
        function generate_full_report(obj)
            %GENERATE_FULL_REPORT Generate complete analysis report
            %
            % Creates:
            %   - Summary tables (CSV and MAT)
            %   - Performance visualizations
            %   - Best configuration analysis
            %   - Text summary report
            
            if obj.verbose
                fprintf('\n=== GENERATING COMPREHENSIVE REPORT ===\n');
            end
            
            % 1. Spectrum-level summary
            if obj.verbose
                fprintf('\n1. Computing spectrum-level summary...\n');
            end
            summary_spectrum = obj.aggregator.summarize('Level', 'spectrum');
            obj.save_summary(summary_spectrum, 'spectrum_level_summary.mat');
            
            % 2. Patient-level summary
            if obj.verbose
                fprintf('2. Computing patient-level summary...\n');
            end
            summary_patient = obj.aggregator.summarize('Level', 'patient');
            obj.save_summary(summary_patient, 'patient_level_summary.mat');
            
            % 3. Export tables
            if obj.verbose
                fprintf('3. Exporting summary tables...\n');
            end
            obj.export_tables(summary_spectrum, summary_patient);
            
            % 4. Find best configurations
            if obj.verbose
                fprintf('4. Identifying best configurations...\n');
            end
            obj.analyze_best_configurations(summary_spectrum, summary_patient);
            
            % 5. Generate visualizations
            if obj.verbose
                fprintf('5. Creating visualizations...\n');
            end
            obj.create_visualizations(summary_spectrum, summary_patient);
            
            % 6. Generate text report
            if obj.verbose
                fprintf('6. Writing text report...\n');
            end
            obj.write_text_report(summary_spectrum, summary_patient);
            
            if obj.verbose
                fprintf('\n=== REPORT GENERATION COMPLETE ===\n');
                fprintf('Output directory: %s\n', obj.output_dir);
            end
        end
        
        function save_summary(obj, summary, filename)
            %SAVE_SUMMARY Save summary structure to MAT file
            
            filepath = fullfile(obj.output_dir, filename);
            save(filepath, 'summary', '-v7.3');
            
            if obj.verbose
                fprintf('  Saved: %s\n', filename);
            end
        end
        
        function export_tables(obj, summary_spectrum, summary_patient)
            %EXPORT_TABLES Export summary tables to CSV
            
            % Spectrum-level table
            tbl_spectrum = obj.aggregator.to_table('Level', 'spectrum');
            spectrum_file = fullfile(obj.output_dir, 'spectrum_level_results.csv');
            writetable(tbl_spectrum, spectrum_file);
            
            % Patient-level table
            tbl_patient = obj.aggregator.to_table('Level', 'patient');
            patient_file = fullfile(obj.output_dir, 'patient_level_results.csv');
            writetable(tbl_patient, patient_file);
            
            if obj.verbose
                fprintf('  Exported: spectrum_level_results.csv\n');
                fprintf('  Exported: patient_level_results.csv\n');
            end
        end
        
        function analyze_best_configurations(obj, summary_spectrum, summary_patient)
            %ANALYZE_BEST_CONFIGURATIONS Find and save best configurations
            
            metrics_to_optimize = {'accuracy', 'macro_f1', 'auc'};
            
            best_configs = struct();
            
            for i = 1:length(metrics_to_optimize)
                metric = metrics_to_optimize{i};
                
                % Spectrum-level best
                best_spectrum = obj.aggregator.get_best_configuration(metric, 'Level', 'spectrum');
                best_configs.spectrum.(metric) = best_spectrum;
                
                % Patient-level best
                best_patient = obj.aggregator.get_best_configuration(metric, 'Level', 'patient');
                best_configs.patient.(metric) = best_patient;
            end
            
            % Save best configurations
            filepath = fullfile(obj.output_dir, 'best_configurations.mat');
            save(filepath, 'best_configs', '-v7.3');
            
            if obj.verbose
                fprintf('  Saved: best_configurations.mat\n');
            end
        end
        
        function create_visualizations(obj, summary_spectrum, summary_patient)
            %CREATE_VISUALIZATIONS Generate all plots
            
            % Performance heatmaps
            obj.visualizer.plot_performance_heatmap(summary_spectrum, 'accuracy');
            obj.visualizer.plot_performance_heatmap(summary_spectrum, 'macro_f1');
            obj.visualizer.plot_performance_heatmap(summary_spectrum, 'auc');
            
            % Classifier comparisons
            comparison_acc = obj.aggregator.compare_classifiers('Metric', 'accuracy', 'Level', 'spectrum');
            obj.visualizer.plot_classifier_comparison(comparison_acc);
            
            comparison_f1 = obj.aggregator.compare_classifiers('Metric', 'macro_f1', 'Level', 'spectrum');
            obj.visualizer.plot_classifier_comparison(comparison_f1);
            
            % Permutation comparisons
            obj.visualizer.plot_permutation_comparison(summary_spectrum, 'accuracy');
            obj.visualizer.plot_permutation_comparison(summary_spectrum, 'macro_f1');
            
            % Boxplots
            obj.visualizer.plot_metric_boxplots(summary_spectrum, 'accuracy');
            obj.visualizer.plot_metric_boxplots(summary_spectrum, 'macro_f1');
            
            if obj.verbose
                fprintf('  Generated visualizations in plots/ subdirectory\n');
            end
        end
        
        function write_text_report(obj, summary_spectrum, summary_patient)
            %WRITE_TEXT_REPORT Generate human-readable text summary
            
            filepath = fullfile(obj.output_dir, 'analysis_summary.txt');
            fid = fopen(filepath, 'w');
            
            if fid == -1
                error('Could not open file for writing: %s', filepath);
            end
            
            try
                % Header
                fprintf(fid, '========================================\n');
                fprintf(fid, 'MENINGIOMA FTIR CLASSIFICATION REPORT\n');
                fprintf(fid, '========================================\n');
                fprintf(fid, 'Generated: %s\n\n', datestr(now));
                
                % Dataset information
                fprintf(fid, 'DATASET INFORMATION\n');
                fprintf(fid, '-------------------\n');
                fprintf(fid, 'Total samples: %d\n', obj.cv_results.n_samples);
                fprintf(fid, 'Total patients: %d\n', obj.cv_results.n_patients);
                fprintf(fid, 'Features: %d\n', obj.cv_results.n_features);
                fprintf(fid, 'Preprocessing permutations: %d\n', obj.cv_results.n_permutations);
                fprintf(fid, 'Classifiers: %d\n', obj.cv_results.n_classifiers);
                fprintf(fid, 'CV folds: %d\n', obj.cv_results.n_folds);
                fprintf(fid, 'CV repeats: %d\n\n', obj.cv_results.n_repeats);
                
                % Best configurations - spectrum level
                fprintf(fid, 'BEST CONFIGURATIONS (SPECTRUM-LEVEL)\n');
                fprintf(fid, '------------------------------------\n');
                obj.write_best_config_section(fid, summary_spectrum, 'accuracy');
                obj.write_best_config_section(fid, summary_spectrum, 'macro_f1');
                obj.write_best_config_section(fid, summary_spectrum, 'auc');
                fprintf(fid, '\n');
                
                % Best configurations - patient level
                fprintf(fid, 'BEST CONFIGURATIONS (PATIENT-LEVEL)\n');
                fprintf(fid, '-----------------------------------\n');
                obj.write_best_config_section(fid, summary_patient, 'accuracy');
                obj.write_best_config_section(fid, summary_patient, 'macro_f1');
                obj.write_best_config_section(fid, summary_patient, 'auc');
                fprintf(fid, '\n');
                
                % Full results table - spectrum level
                fprintf(fid, 'COMPLETE RESULTS (SPECTRUM-LEVEL)\n');
                fprintf(fid, '---------------------------------\n');
                obj.write_results_table(fid, summary_spectrum);
                fprintf(fid, '\n');
                
                % Full results table - patient level
                fprintf(fid, 'COMPLETE RESULTS (PATIENT-LEVEL)\n');
                fprintf(fid, '--------------------------------\n');
                obj.write_results_table(fid, summary_patient);
                
                fclose(fid);
                
                if obj.verbose
                    fprintf('  Saved: analysis_summary.txt\n');
                end
            catch ME
                fclose(fid);
                rethrow(ME);
            end
        end
        
        function write_best_config_section(obj, fid, summary, metric_name)
            %WRITE_BEST_CONFIG_SECTION Write best configuration for metric
            
            best = obj.aggregator.get_best_configuration(metric_name, 'Level', summary.level);
            
            fprintf(fid, '\nBest for %s:\n', upper(metric_name));
            fprintf(fid, '  Permutation: %s\n', best.permutation_id);
            fprintf(fid, '  Classifier: %s\n', best.classifier_name);
            fprintf(fid, '  %s: %.4f ± %.4f\n', metric_name, ...
                best.best_value, best.std_metrics.(metric_name));
            fprintf(fid, '  Accuracy: %.4f ± %.4f\n', ...
                best.mean_metrics.accuracy, best.std_metrics.accuracy);
            fprintf(fid, '  F1: %.4f ± %.4f\n', ...
                best.mean_metrics.macro_f1, best.std_metrics.macro_f1);
        end
        
        function write_results_table(obj, fid, summary)
            %WRITE_RESULTS_TABLE Write formatted results table
            
            fprintf(fid, '\n%-15s %-15s %10s %10s %10s %10s %10s %10s\n', ...
                'Permutation', 'Classifier', 'Acc', 'Acc_Std', 'F1', 'F1_Std', 'AUC', 'AUC_Std');
            fprintf(fid, '%s\n', repmat('-', 1, 115));
            
            for p = 1:size(summary.configurations, 1)
                for c = 1:size(summary.configurations, 2)
                    config = summary.configurations{p, c};
                    
                    fprintf(fid, '%-15s %-15s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n', ...
                        config.permutation_id, config.classifier_name, ...
                        config.mean_metrics.accuracy, config.std_metrics.accuracy, ...
                        config.mean_metrics.macro_f1, config.std_metrics.macro_f1, ...
                        config.mean_metrics.auc, config.std_metrics.auc);
                end
            end
        end
    end
end
