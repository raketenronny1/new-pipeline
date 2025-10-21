%% SIMPLIFIED PATIENT-WISE PIPELINE
% Streamlined pipeline that works directly with data tables
%
% Key improvements:
% - No intermediate files (works directly with data tables)
% - Proper Patient_ID stratification
% - Integrated PCA within CV (no separate step)
% - Cleaner code structure

function run_pipeline_direct(perform_qc)
    % Run the complete pipeline with direct data access
    %
    % INPUT:
    %   perform_qc: (optional) boolean, run quality control first
    
    if nargin < 1, perform_qc = false; end
    
    %% Setup
    addpath(fileparts(mfilename('fullpath')));
    cfg = config();
    
    fprintf('\n═══════════════════════════════════════════════════════════\n');
    fprintf('  MENINGIOMA FTIR CLASSIFICATION PIPELINE (DIRECT)\n');
    fprintf('═══════════════════════════════════════════════════════════\n');
    fprintf('Working directory: %s\n', pwd);
    fprintf('Data directory: %s\n', cfg.paths.data);
    fprintf('Results directory: %s\n', cfg.paths.results);
    fprintf('═══════════════════════════════════════════════════════════\n\n');
    
    %% Phase 0: Quality Control (Optional)
    if perform_qc
        fprintf('\n=== PHASE 0: QUALITY CONTROL ===\n');
        qc_file = fullfile(cfg.paths.qc, 'qc_flags.mat');
        
        if exist(qc_file, 'file')
            fprintf('QC results already exist: %s\n', qc_file);
            fprintf('Skipping QC analysis.\n');
        else
            fprintf('Running quality control analysis...\n');
            quality_control_analysis(cfg);
            fprintf('✓ QC complete\n');
        end
    else
        fprintf('\n=== PHASE 0: QUALITY CONTROL SKIPPED ===\n');
        fprintf('Note: All spectra will be used (no QC filtering)\n');
    end
    
    %% Phase 1: Load Data (Direct Access)
    fprintf('\n=== PHASE 1: DATA LOADING ===\n');
    fprintf('Loading data directly from tables...\n');
    
    data = load_data_direct(cfg);
    
    fprintf('✓ Data loaded successfully\n');
    fprintf('  Training: %d samples, %d patients, %d spectra\n', ...
            data.train.n_samples, ...
            length(unique(data.train.patient_id)), ...
            data.train.total_spectra);
    fprintf('  Test: %d samples, %d patients, %d spectra\n', ...
            data.test.n_samples, ...
            length(unique(data.test.patient_id)), ...
            data.test.total_spectra);
    
    %% Phase 2: Cross-Validation (with integrated PCA)
    fprintf('\n=== PHASE 2: PATIENT-WISE CROSS-VALIDATION ===\n');
    fprintf('Key principles:\n');
    fprintf('  • Stratification by Patient_ID (prevents data leakage)\n');
    fprintf('  • Each Diss_ID treated as independent sample\n');
    fprintf('  • Individual spectrum prediction\n');
    fprintf('  • Majority voting aggregation per sample\n');
    fprintf('  • PCA applied within each fold\n\n');
    
    cv_results = run_patientwise_cv_direct(data, cfg);
    
    fprintf('\n✓ Cross-validation complete\n');
    
    %% Phase 3: Generate Report
    fprintf('\n=== PHASE 3: RESULTS SUMMARY ===\n');
    
    % Display results for each classifier
    classifier_names = fieldnames(cv_results);
    classifier_names = classifier_names(~strcmp(classifier_names, 'metadata'));
    
    fprintf('\n');
    fprintf('┌─────────────────┬──────────┬─────────────┬─────────────┬─────────────┐\n');
    fprintf('│ Classifier      │ Accuracy │ Sensitivity │ Specificity │     AUC     │\n');
    fprintf('├─────────────────┼──────────┼─────────────┼─────────────┼─────────────┤\n');
    
    for i = 1:length(classifier_names)
        clf_name = classifier_names{i};
        m = cv_results.(clf_name).metrics;
        
        fprintf('│ %-15s │ %.3f±%.2f │   %.3f±%.2f  │   %.3f±%.2f  │  %.3f±%.2f  │\n', ...
                clf_name, ...
                m.accuracy_mean, m.accuracy_std, ...
                m.sensitivity_mean, m.sensitivity_std, ...
                m.specificity_mean, m.specificity_std, ...
                m.auc_mean, m.auc_std);
    end
    
    fprintf('└─────────────────┴──────────┴─────────────┴─────────────┴─────────────┘\n');
    
    %% Find Best Classifier
    best_clf = '';
    best_acc = 0;
    
    for i = 1:length(classifier_names)
        clf_name = classifier_names{i};
        acc = cv_results.(clf_name).metrics.accuracy_mean;
        if acc > best_acc
            best_acc = acc;
            best_clf = clf_name;
        end
    end
    
    fprintf('\n🏆 Best classifier: %s (Accuracy: %.3f)\n', best_clf, best_acc);
    
    %% Export Detailed Results
    fprintf('\n=== EXPORTING RESULTS ===\n');
    
    % Save detailed results
    results_file = fullfile(cfg.paths.results, 'cv_results_direct.mat');
    fprintf('Saving to: %s\n', results_file);
    
    % Export to Excel for easy inspection
    export_results_to_excel(cv_results, cfg.paths.results);
    
    %% Final Summary
    fprintf('\n═══════════════════════════════════════════════════════════\n');
    fprintf('  PIPELINE COMPLETE!\n');
    fprintf('═══════════════════════════════════════════════════════════\n');
    fprintf('Results directory: %s\n', cfg.paths.results);
    fprintf('\nGenerated files:\n');
    fprintf('  • cv_results_direct.mat - Full results structure\n');
    fprintf('  • cv_predictions.xlsx - Detailed predictions per sample\n');
    fprintf('  • cv_summary.txt - Performance summary\n');
    fprintf('═══════════════════════════════════════════════════════════\n\n');
    
    % Return results
    assignin('base', 'cv_results', cv_results);
    assignin('base', 'data', data);
    
    fprintf('Results also available in workspace as:\n');
    fprintf('  • cv_results (structure)\n');
    fprintf('  • data (structure)\n\n');
end


%% Helper: Export results to Excel
function export_results_to_excel(cv_results, results_dir)
    classifier_names = fieldnames(cv_results);
    classifier_names = classifier_names(~strcmp(classifier_names, 'metadata'));
    
    % Create summary file
    summary_file = fullfile(results_dir, 'cv_summary.txt');
    fid = fopen(summary_file, 'w');
    
    fprintf(fid, 'MENINGIOMA FTIR CLASSIFICATION - CV RESULTS\n');
    fprintf(fid, '==========================================\n\n');
    
    for i = 1:length(classifier_names)
        clf_name = classifier_names{i};
        m = cv_results.(clf_name).metrics;
        
        fprintf(fid, '%s:\n', clf_name);
        fprintf(fid, '  Accuracy:    %.3f ± %.3f\n', m.accuracy_mean, m.accuracy_std);
        fprintf(fid, '  Sensitivity: %.3f ± %.3f\n', m.sensitivity_mean, m.sensitivity_std);
        fprintf(fid, '  Specificity: %.3f ± %.3f\n', m.specificity_mean, m.specificity_std);
        fprintf(fid, '  Precision:   %.3f\n', m.precision_mean);
        fprintf(fid, '  F1-Score:    %.3f\n', m.f1_mean);
        fprintf(fid, '  AUC:         %.3f ± %.3f\n', m.auc_mean, m.auc_std);
        fprintf(fid, '\n');
    end
    
    fclose(fid);
    fprintf('  Summary saved to: cv_summary.txt\n');
    
    % Create detailed predictions Excel file
    excel_file = fullfile(results_dir, 'cv_predictions.xlsx');
    
    for i = 1:length(classifier_names)
        clf_name = classifier_names{i};
        res = cv_results.(clf_name);
        
        % Create table
        T = table(res.sample_ids, res.patient_ids, res.sample_true, res.sample_predictions, ...
                  res.fold_info(:,1), res.fold_info(:,2), ...
                  'VariableNames', {'Sample_ID', 'Patient_ID', 'True_Label', 'Predicted', 'Repeat', 'Fold'});
        
        % Add correctness column
        T.Correct = (T.True_Label == T.Predicted);
        
        % Write to Excel sheet
        writetable(T, excel_file, 'Sheet', clf_name);
    end
    
    fprintf('  Detailed predictions saved to: cv_predictions.xlsx\n');
end
