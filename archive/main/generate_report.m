%% PHASE 6: GENERATE COMPREHENSIVE REPORT
% This script generates a complete report of the analysis results

function generate_report(cfg)
    % Input validation
    if ~isstruct(cfg) || ~isfield(cfg.paths, 'results') || ~isfield(cfg.paths, 'qc')
        error('Invalid cfg structure. Must contain paths.results and paths.qc');
    end

    try
        % Pre-check file existence to fail fast
        required_files = {
            fullfile(cfg.paths.results, 'cv_performance.csv')
            fullfile(cfg.paths.results, 'best_classifier_selection.mat')
            fullfile(cfg.paths.results, 'test_results.mat')
            fullfile(cfg.paths.qc, 'qc_metrics_train.csv')
            fullfile(cfg.paths.qc, 'qc_metrics_test.csv')
        };
        
        for i = 1:length(required_files)
            if ~exist(required_files{i}, 'file')
                error('Required file not found: %s', required_files{i});
            end
        end

        % Load all results efficiently
        [summary_table, best_model_info, test_results] = load_results(cfg);
        [qc_metrics_train, qc_metrics_test] = load_qc_metrics(cfg);

    % Create report file
    report_file = fullfile(cfg.paths.results, 'final_report.md');
    fid = fopen(report_file, 'w');

    try
        % === 1. Executive Summary ===
        write_executive_summary(fid, summary_table, best_model_info, test_results);

        % === 2. Methods Section ===
        write_methods_section(fid, qc_metrics_train, qc_metrics_test);

        % === 3. Results Tables ===
        write_results_tables(fid, summary_table, test_results);

        % === 4. Quality Control Summary ===
        write_qc_summary(fid, qc_metrics_train, qc_metrics_test);

        % === 5. Discussion Points ===
        write_discussion_points(fid, test_results, best_model_info);

        % Save report
        fclose(fid);

        % Convert to PDF/Word if possible
        try_convert_to_pdf_or_docx(report_file);

    catch ME
        if ~isempty(fopen('all'))
            fclose(fid);
        end
        rethrow(ME);
    end
end

function write_executive_summary(fid, summary_table, best_model_info, test_results)
    % Pre-format strings for better performance
    metrics = {
        'Mean Balanced Accuracy' summary_table.Mean_Accuracy summary_table.SD_Accuracy
        'Mean Sensitivity (WHO-3)' summary_table.Mean_Sensitivity_WHO3 summary_table.SD_Sensitivity_WHO3
        'Mean Specificity (WHO-1)' summary_table.Mean_Specificity_WHO1 summary_table.SD_Specificity_WHO1
        'Mean F2-Score (WHO-3)' summary_table.Mean_F2_WHO3 summary_table.SD_F2_WHO3
    };
    
    % Build report sections efficiently using string arrays
    sections = [
        "# Performance Summary\n\n"
        "## Training Set Cross-Validation:\n"
    ];
    
    % Generate metric lines efficiently
    metric_lines = arrayfun(@(i) sprintf('- %s: %.1f%% ± %.1f%%\n', ...
        metrics{i,1}, metrics{i,2}*100, metrics{i,3}*100), ...
        1:size(metrics,1), 'UniformOutput', false);
    
    % Write all at once
    fprintf(fid, strjoin([sections(:); metric_lines(:)], ''));
    % ... add other metrics
    
    % Test results
    fprintf(fid, '\n## Test Set Performance:\n');
    fprintf(fid, '- Balanced Accuracy: %.1f%%\n', ...
            test_results.metrics.balanced_accuracy * 100);
    fprintf(fid, '- Sensitivity (WHO-3): %.1f%%\n', ...
            test_results.metrics.sensitivity * 100);
    % ... add other metrics
    
    % Train-test gap analysis
    fprintf(fid, '\n## Train-Test Gap Analysis:\n');
    % Calculate and report gaps
end

function write_methods_section(fid, qc_metrics_train, qc_metrics_test)
    fprintf(fid, '\n# Methods\n\n');
    
    % Dataset description
    fprintf(fid, '## Dataset and Quality Control\n\n');
    % Write dataset details
    
    % QC methodology
    fprintf(fid, '## Quality Control Methodology\n\n');
    % Document QC steps
    
    % Feature extraction
    fprintf(fid, '## Feature Extraction and Model Development\n\n');
    % Document PCA and modeling approach
end

function write_results_tables(fid, summary_table, test_results)
    fprintf(fid, '\n# Results\n\n');
    
    % Table 1: CV Performance
    fprintf(fid, '## Cross-Validation Performance\n\n');
    % Format and write CV results table
    
    % Table 2: Test Performance
    fprintf(fid, '## Test Set Performance\n\n');
    % Format and write test results table
end

function write_qc_summary(fid, qc_metrics_train, qc_metrics_test)
    fprintf(fid, '\n# Quality Control Results\n\n');
    
    % Spectrum-level statistics
    fprintf(fid, '## Spectrum-Level Filtering\n');
    % Calculate and report QC statistics
    
    % Sample-level assessment
    fprintf(fid, '## Sample-Level Assessment\n');
    % Report sample exclusions and reasons
end

function write_discussion_points(fid, test_results, best_model_info)
    fprintf(fid, '\n# Discussion\n\n');
    
    % Model performance
    fprintf(fid, '## Model Performance\n');
    % Discuss performance in context
    
    % Generalization
    fprintf(fid, '## Generalization Assessment\n');
    % Analyze train-test gap
    
    % Clinical relevance
    fprintf(fid, '## Clinical Relevance\n');
    % Discuss implications
    
    % Limitations
    fprintf(fid, '## Limitations\n');
    % List key limitations
    
    % Strengths
    fprintf(fid, '## Strengths\n');
    % Highlight key strengths
end

function [summary_table, best_model_info, test_results] = load_results(cfg)
    % Load results files efficiently
    summary_table = readtable(fullfile(cfg.paths.results, 'cv_performance.csv'));
    
    tmp = load(fullfile(cfg.paths.results, 'best_classifier_selection.mat'));
    best_model_info = tmp.best_model_info;
    
    tmp = load(fullfile(cfg.paths.results, 'test_results.mat'));
    test_results = tmp.test_results;
end

function [qc_metrics_train, qc_metrics_test] = load_qc_metrics(cfg)
    % Load QC metrics with optimized table reading
    opts = detectImportOptions(fullfile(cfg.paths.qc, 'qc_metrics_train.csv'));
    opts.PreserveVariableNames = true;
    qc_metrics_train = readtable(fullfile(cfg.paths.qc, 'qc_metrics_train.csv'), opts);
    qc_metrics_test = readtable(fullfile(cfg.paths.qc, 'qc_metrics_test.csv'), opts);
end

function try_convert_to_pdf_or_docx(markdown_file)
    % Try to convert markdown to PDF/Word if pandoc is available
    [status, ~] = system('where pandoc');
    if status == 0
        % Create temporary directory for PDF generation
        temp_dir = fullfile(fileparts(markdown_file), 'temp_pandoc');
        if ~exist(temp_dir, 'dir')
            mkdir(temp_dir);
        end
        
        try
            % Convert to PDF with better error handling
            pdf_cmd = sprintf('pandoc "%s" -o "%s" --pdf-engine=xelatex --resource-path="%s"', ...
                markdown_file, strrep(markdown_file, '.md', '.pdf'), temp_dir);
            [status, result] = system(pdf_cmd);
            if status ~= 0
                warning('PDF conversion failed: %s', result);
            end
            
            % Convert to Word
            docx_cmd = sprintf('pandoc "%s" -o "%s"', ...
                markdown_file, strrep(markdown_file, '.md', '.docx'));
            [status, result] = system(docx_cmd);
            if status ~= 0
                warning('Word conversion failed: %s', result);
            end
        finally
            % Cleanup
            if exist(temp_dir, 'dir')
                rmdir(temp_dir, 's');
            end
        end
    else
        warning('Pandoc not found. Only markdown report generated.');
    end
end