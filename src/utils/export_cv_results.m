%% EXPORT_CV_RESULTS - Export cross-validation results to files
%
% Exports CV results to Excel and text summary formats for analysis and reporting.
%
% SYNTAX:
%   export_cv_results(cv_results, results_dir)
%   export_cv_results(cv_results, results_dir, 'Pipeline', pipeline_name)
%
% INPUTS:
%   cv_results  - CV results structure from run_patientwise_cv_direct
%   results_dir - Directory path for output files
%
% OPTIONAL NAME-VALUE PAIRS:
%   'Pipeline' - Pipeline description (default: 'EDA-based outlier removal')
%   'Verbose'  - Display progress (default: true)
%
% OUTPUTS:
%   Creates two files in results_dir:
%   - cv_summary.txt: Text summary with mean ± std metrics for all classifiers
%   - cv_predictions.xlsx: Excel file with detailed predictions (one sheet per classifier)
%
% EXAMPLES:
%   % Basic usage
%   export_cv_results(cv_results, 'results/');
%
%   % With custom pipeline description
%   export_cv_results(cv_results, 'results/', 'Pipeline', 'QC-based filtering');
%
% See also: run_pipeline, run_patientwise_cv_direct, export_test_results

function export_cv_results(cv_results, results_dir, varargin)
    %% Parse inputs
    p = inputParser;
    addRequired(p, 'cv_results', @isstruct);
    addRequired(p, 'results_dir', @ischar);
    addParameter(p, 'Pipeline', 'EDA-based outlier removal', @ischar);
    addParameter(p, 'Verbose', true, @islogical);
    parse(p, cv_results, results_dir, varargin{:});
    
    pipeline_desc = p.Results.Pipeline;
    verbose = p.Results.Verbose;
    
    %% Get classifier names
    classifier_names = fieldnames(cv_results);
    classifier_names = classifier_names(~strcmp(classifier_names, 'metadata'));
    
    if verbose
        fprintf('Exporting CV results...\n');
    end
    
    %% Create text summary
    summary_file = fullfile(results_dir, 'cv_summary.txt');
    fid = fopen(summary_file, 'w');
    
    fprintf(fid, 'MENINGIOMA FTIR CLASSIFICATION - CROSS-VALIDATION RESULTS\n');
    fprintf(fid, '=========================================================\n\n');
    fprintf(fid, 'Pipeline: %s\n', pipeline_desc);
    fprintf(fid, 'CV: Patient-stratified k-fold\n');
    
    % Check if PCA model was used
    if isfield(cv_results, 'metadata') && isfield(cv_results.metadata, 'pca_components')
        fprintf(fid, 'PCA: %d components (used for LDA only)\n\n', ...
                cv_results.metadata.pca_components);
    else
        fprintf(fid, '\n');
    end
    
    % Write metrics for each classifier
    for i = 1:length(classifier_names)
        clf_name = classifier_names{i};
        m = cv_results.(clf_name).metrics;
        
        fprintf(fid, '%s:\n', clf_name);
        fprintf(fid, '  Accuracy:    %.3f ± %.3f\n', m.accuracy_mean, m.accuracy_std);
        fprintf(fid, '  Sensitivity: %.3f ± %.3f (WHO-3 detection)\n', ...
                m.sensitivity_mean, m.sensitivity_std);
        fprintf(fid, '  Specificity: %.3f ± %.3f (WHO-1 detection)\n', ...
                m.specificity_mean, m.specificity_std);
        fprintf(fid, '  Precision:   %.3f ± %.3f\n', m.precision_mean, m.precision_std);
        fprintf(fid, '  F1-Score:    %.3f ± %.3f\n', m.f1_mean, m.f1_std);
        fprintf(fid, '  AUC:         %.3f ± %.3f\n', m.auc_mean, m.auc_std);
        fprintf(fid, '\n');
    end
    
    fclose(fid);
    
    if verbose
        fprintf('  Summary saved: %s\n', summary_file);
    end
    
    %% Create Excel file with detailed predictions
    excel_file = fullfile(results_dir, 'cv_predictions.xlsx');
    
    for i = 1:length(classifier_names)
        clf_name = classifier_names{i};
        res = cv_results.(clf_name);
        
        % Create table with predictions
        T = table();
        
        if isfield(res, 'sample_ids')
            T.Sample_ID = res.sample_ids;
        end
        
        if isfield(res, 'patient_ids')
            T.Patient_ID = res.patient_ids;
        end
        
        T.True_Label = res.sample_true;
        T.Predicted_Label = res.sample_predictions;
        T.Correct = (T.True_Label == T.Predicted_Label);
        
        % Add confidence if available
        if isfield(res, 'sample_confidence')
            T.Confidence = res.sample_confidence;
        end
        
        % Write to Excel sheet
        writetable(T, excel_file, 'Sheet', clf_name);
    end
    
    if verbose
        fprintf('  Predictions saved: %s\n', excel_file);
    end
end
