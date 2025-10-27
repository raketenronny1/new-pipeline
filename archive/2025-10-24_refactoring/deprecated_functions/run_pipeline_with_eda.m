%% STREAMLINED ML PIPELINE WITH EDA-BASED QC
% Complete pipeline integrating EDA outlier detection with classification
%
% WORKFLOW:
%   1. EDA performs PCA and T²-Q outlier detection on training data
%   2. Outliers are removed from training set
%   3. EDA PCA model (15 PCs) is used for LDA classifier
%   4. Patient-stratified cross-validation with all classifiers
%   5. Test set evaluation (no outlier removal)
%
% KEY IMPROVEMENTS:
%   - Single PCA model from EDA used consistently
%   - Outlier detection based on statistical criteria (T², Q)
%   - No redundant Mahalanobis distance calculation
%   - Streamlined data flow

function run_pipeline_with_eda()
    %% Setup
    addpath(fileparts(mfilename('fullpath')));
    cfg = config();
    
    % Update paths to include EDA results
    cfg.paths.eda = 'results/eda';
    
    fprintf('\n═══════════════════════════════════════════════════════════\n');
    fprintf('  STREAMLINED ML PIPELINE WITH EDA-BASED QC\n');
    fprintf('═══════════════════════════════════════════════════════════\n');
    fprintf('Working directory: %s\n', pwd);
    fprintf('Data directory: %s\n', cfg.paths.data);
    fprintf('EDA directory: %s\n', cfg.paths.eda);
    fprintf('Results directory: %s\n', cfg.paths.results);
    fprintf('═══════════════════════════════════════════════════════════\n\n');
    
    %% Phase 1: Run EDA if not already done
    fprintf('\n=== PHASE 1: EXPLORATORY DATA ANALYSIS ===\n');
    eda_file = fullfile(cfg.paths.eda, 'eda_results_PP1.mat');
    
    if exist(eda_file, 'file')
        fprintf('EDA results already exist: %s\n', eda_file);
        fprintf('Skipping EDA analysis.\n');
        fprintf('(Delete file and rerun if you want fresh EDA)\n');
    else
        fprintf('Running EDA with outlier detection...\n');
        fprintf('This will:\n');
        fprintf('  • Compute PCA on training data (WHO-1 & WHO-3)\n');
        fprintf('  • Detect outliers using T² and Q statistics\n');
        fprintf('  • Generate visualizations\n');
        fprintf('  • Save PCA model (15 components) for downstream use\n\n');
        
        % Run EDA
        run_full_eda();
        
        fprintf('\n✓ EDA complete\n');
    end
    
    %% Phase 2: Load Data with EDA Filtering
    fprintf('\n=== PHASE 2: DATA LOADING WITH EDA FILTERING ===\n');
    fprintf('Loading data with EDA-based outlier removal...\n');
    
    data = load_data_with_eda(cfg);
    
    fprintf('✓ Data loaded successfully\n');
    fprintf('  Training: %d samples, %d patients, %d spectra (after outlier removal)\n', ...
            data.train.n_samples, ...
            length(unique(data.train.patient_id)), ...
            data.train.total_spectra);
    fprintf('  Test: %d samples, %d patients, %d spectra (no filtering)\n', ...
            data.test.n_samples, ...
            length(unique(data.test.patient_id)), ...
            data.test.total_spectra);
    fprintf('  PCA Model: %d components (%.1f%% variance)\n', ...
            data.pca_model.n_comp, data.pca_model.total_variance);
    
    %% Phase 3: Patient-Wise Cross-Validation
    fprintf('\n=== PHASE 3: PATIENT-WISE CROSS-VALIDATION ===\n');
    fprintf('Key principles:\n');
    fprintf('  • Stratification by Patient_ID (prevents data leakage)\n');
    fprintf('  • Each Diss_ID treated as independent sample\n');
    fprintf('  • Individual spectrum prediction\n');
    fprintf('  • Majority voting aggregation per sample\n');
    fprintf('  • LDA uses EDA PCA model (15 PCs)\n');
    fprintf('  • Other classifiers use standardized spectra\n\n');
    
    cv_results = run_patientwise_cv_direct(data, cfg);
    
    fprintf('\n✓ Cross-validation complete\n');
    
    %% Phase 4: Results Summary
    fprintf('\n=== PHASE 4: RESULTS SUMMARY ===\n');
    
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
    
    %% Export Results
    fprintf('\n=== EXPORTING RESULTS ===\n');
    
    % Save detailed results
    results_file = fullfile(cfg.paths.results, 'cv_results_eda_pipeline.mat');
    save(results_file, 'cv_results', 'data', '-v7.3');
    fprintf('Saved to: %s\n', results_file);
    
    % Export to Excel
    export_results_to_excel(cv_results, cfg.paths.results);
    
    %% Final Summary
    fprintf('\n═══════════════════════════════════════════════════════════\n');
    fprintf('  PIPELINE COMPLETE!\n');
    fprintf('═══════════════════════════════════════════════════════════\n');
    fprintf('EDA directory: %s\n', cfg.paths.eda);
    fprintf('Results directory: %s\n', cfg.paths.results);
    fprintf('\nGenerated files:\n');
    fprintf('  EDA:\n');
    fprintf('    • eda_results_PP1.mat - PCA model and outlier flags\n');
    fprintf('    • Visualization plots (01_*.png, 02_*.png, etc.)\n');
    fprintf('  CV Results:\n');
    fprintf('    • cv_results_eda_pipeline.mat - Full results\n');
    fprintf('    • cv_predictions.xlsx - Detailed predictions\n');
    fprintf('    • cv_summary.txt - Performance summary\n');
    fprintf('═══════════════════════════════════════════════════════════\n\n');
    
    % Return results
    assignin('base', 'cv_results', cv_results);
    assignin('base', 'data', data);
    
    fprintf('Results also available in workspace as:\n');
    fprintf('  • cv_results (structure)\n');
    fprintf('  • data (structure with PCA model)\n\n');
end


%% Helper: Export results to Excel
function export_results_to_excel(cv_results, results_dir)
    classifier_names = fieldnames(cv_results);
    classifier_names = classifier_names(~strcmp(classifier_names, 'metadata'));
    
    % Create summary file
    summary_file = fullfile(results_dir, 'cv_summary.txt');
    fid = fopen(summary_file, 'w');
    
    fprintf(fid, 'MENINGIOMA FTIR CLASSIFICATION - CV RESULTS (EDA Pipeline)\n');
    fprintf(fid, '==========================================================\n\n');
    fprintf(fid, 'Pipeline: EDA-based outlier removal + Patient-stratified CV\n');
    fprintf(fid, 'PCA: 15 components from EDA (used for LDA only)\n\n');
    
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
        T = table();
        T.True_Labels = res.sample_true;
        T.Predicted_Labels = res.sample_predictions;
        T.Sample_IDs = res.sample_ids;
        T.Patient_IDs = res.patient_ids;
        T.Correct = (T.True_Labels == T.Predicted_Labels);
        
        writetable(T, excel_file, 'Sheet', clf_name);
    end
    
    fprintf('  Detailed predictions saved to: cv_predictions.xlsx\n');
end
