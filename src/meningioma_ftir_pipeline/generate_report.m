%% PHASE 6: GENERATE COMPREHENSIVE REPORT
% This script generates a complete report of the analysis results

function generate_report(cfg)
    % Load all results
    load(fullfile(cfg.paths.results, 'cv_performance.csv'), 'summary_table');
    load(fullfile(cfg.paths.results, 'best_classifier_selection.mat'), 'best_model_info');
    load(fullfile(cfg.paths.results, 'test_results.mat'), 'test_results');
    qc_metrics_train = readtable(fullfile(cfg.paths.qc, 'qc_metrics_train.csv'));
    qc_metrics_test = readtable(fullfile(cfg.paths.qc, 'qc_metrics_test.csv'));

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
    fprintf(fid, '# Performance Summary\n\n');
    
    % CV results
    fprintf(fid, '## Training Set Cross-Validation:\n');
    fprintf(fid, '- Mean Balanced Accuracy: %.1f%% ± %.1f%%\n', ...
            summary_table.Mean_Accuracy * 100, ...
            summary_table.SD_Accuracy * 100);
    fprintf(fid, '- Mean Sensitivity (WHO-3): %.1f%% ± %.1f%%\n', ...
            summary_table.Mean_Sensitivity_WHO3 * 100, ...
            summary_table.SD_Sensitivity_WHO3 * 100);
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

function try_convert_to_pdf_or_docx(markdown_file)
    % Try to convert markdown to PDF/Word if pandoc is available
    [status, ~] = system('pandoc --version');
    if status == 0
        % Convert to PDF
        system(sprintf('pandoc %s -o %s --pdf-engine=xelatex', ...
               markdown_file, strrep(markdown_file, '.md', '.pdf')));
        
        % Convert to Word
        system(sprintf('pandoc %s -o %s', ...
               markdown_file, strrep(markdown_file, '.md', '.docx')));
    else
        warning('Pandoc not found. Only markdown report generated.');
    end
end