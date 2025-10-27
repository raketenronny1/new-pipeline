function exportDetailedResults(cvResults, patientData, output_dir, filename)
    % Exports comprehensive results to Excel for clinical review
    %
    % INPUT:
    %   cvResults: struct array with CV results
    %   patientData: original patient data struct
    %   output_dir: directory to save files
    %   filename: base filename (default: 'cv_results_detailed')
    
    if nargin < 3 || isempty(output_dir)
        output_dir = 'results/meningioma_ftir_pipeline/';
    end
    
    if nargin < 4
        filename = 'cv_results_detailed';
    end
    
    % Create output directory if it doesn't exist
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Aggregate all patient-level results
    allPatientResults = [];
    allFoldInfo = [];
    for k = 1:length(cvResults)
        if isfield(cvResults(k), 'patientLevelResults')
            nPats = length(cvResults(k).patientLevelResults);
            foldInfo = repmat(k, nPats, 1);
            allPatientResults = [allPatientResults; cvResults(k).patientLevelResults];
            allFoldInfo = [allFoldInfo; foldInfo];
        end
    end
    
    if isempty(allPatientResults)
        warning('No results to export');
        return;
    end
    
    % Create table
    T = struct2table(allPatientResults);
    T.FoldNumber = allFoldInfo;
    
    % Reorder columns for readability
    base_cols = {'FoldNumber', 'patientID', 'trueLabel', 'predictedLabel', ...
                 'isCorrect', 'nSpectra', 'nPredictedWHO1', 'nPredictedWHO3', ...
                 'majorityVoteConfidence', 'meanProbWHO1', 'meanProbWHO3', ...
                 'stdProbWHO1', 'stdProbWHO3', 'predictionEntropy'};
    
    % Only keep columns that exist in T
    cols_to_keep = base_cols(ismember(base_cols, T.Properties.VariableNames));
    T = T(:, cols_to_keep);
    
    % Add clinical interpretation column
    T.InterpretationFlag = cell(height(T), 1);
    for i = 1:height(T)
        if T.majorityVoteConfidence(i) >= 0.85 && T.isCorrect(i)
            T.InterpretationFlag{i} = 'High Confidence - Correct';
        elseif T.majorityVoteConfidence(i) >= 0.85 && ~T.isCorrect(i)
            T.InterpretationFlag{i} = 'High Confidence - INCORRECT (Review!)';
        elseif T.majorityVoteConfidence(i) <= 0.60
            T.InterpretationFlag{i} = 'Low Confidence - Ambiguous Case';
        else
            T.InterpretationFlag{i} = 'Moderate Confidence';
        end
    end
    
    % Export to Excel
    excel_file = fullfile(output_dir, [filename '.xlsx']);
    writetable(T, excel_file, 'Sheet', 'Patient Results');
    fprintf('✓ Detailed results exported to: %s\n', excel_file);
    
    % Also save summary statistics
    summaryFile = fullfile(output_dir, [filename '_summary.txt']);
    fid = fopen(summaryFile, 'w');
    
    fprintf(fid, '═══════════════════════════════════════════════════════════\n');
    fprintf(fid, '  PATIENT-WISE CROSS-VALIDATION SUMMARY\n');
    fprintf(fid, '═══════════════════════════════════════════════════════════\n\n');
    
    fprintf(fid, 'Analysis Date: %s\n\n', datestr(now));
    
    fprintf(fid, 'Number of Folds: %d\n', length(cvResults));
    fprintf(fid, 'Total Patients: %d\n', length(allPatientResults));
    
    if isfield(cvResults(1), 'aggregated')
        agg = cvResults(1).aggregated;
        fprintf(fid, '\n--- Performance Metrics (Mean ± SD) [95%% CI] ---\n');
        fprintf(fid, 'Accuracy:    %.2f%% ± %.2f%% [±%.2f%%]\n', ...
                agg.meanAccuracy*100, agg.stdAccuracy*100, agg.ciAccuracy*100);
        fprintf(fid, 'Sensitivity: %.2f%% ± %.2f%% [±%.2f%%]\n', ...
                agg.meanSensitivity*100, agg.stdSensitivity*100, agg.ciSensitivity*100);
        fprintf(fid, 'Specificity: %.2f%% ± %.2f%% [±%.2f%%]\n', ...
                agg.meanSpecificity*100, agg.stdSpecificity*100, agg.ciSpecificity*100);
        fprintf(fid, 'PPV:         %.2f%% ± %.2f%%\n', ...
                agg.meanPPV*100, agg.stdPPV*100);
        fprintf(fid, 'NPV:         %.2f%% ± %.2f%%\n', ...
                agg.meanNPV*100, agg.stdNPV*100);
        fprintf(fid, 'F1-Score:    %.3f ± %.3f [±%.3f]\n', ...
                agg.meanF1Score, agg.stdF1Score, agg.ciF1Score);
    end
    
    fprintf(fid, '\n--- Clinical Interpretation Distribution ---\n');
    flags = T.InterpretationFlag;
    fprintf(fid, 'High Confidence Correct:     %d (%.1f%%)\n', ...
            sum(strcmp(flags, 'High Confidence - Correct')), ...
            100*sum(strcmp(flags, 'High Confidence - Correct'))/height(T));
    fprintf(fid, 'High Confidence INCORRECT:   %d (%.1f%%) *** REVIEW REQUIRED ***\n', ...
            sum(strcmp(flags, 'High Confidence - INCORRECT (Review!)')), ...
            100*sum(strcmp(flags, 'High Confidence - INCORRECT (Review!)'))/height(T));
    fprintf(fid, 'Low Confidence Ambiguous:    %d (%.1f%%)\n', ...
            sum(strcmp(flags, 'Low Confidence - Ambiguous Case')), ...
            100*sum(strcmp(flags, 'Low Confidence - Ambiguous Case'))/height(T));
    fprintf(fid, 'Moderate Confidence:         %d (%.1f%%)\n', ...
            sum(strcmp(flags, 'Moderate Confidence')), ...
            100*sum(strcmp(flags, 'Moderate Confidence'))/height(T));
    
    fprintf(fid, '\n--- Confidence Metrics ---\n');
    fprintf(fid, 'Mean Confidence: %.3f ± %.3f\n', ...
            mean(T.majorityVoteConfidence), std(T.majorityVoteConfidence));
    fprintf(fid, 'Mean Entropy: %.3f ± %.3f\n', ...
            mean(T.predictionEntropy), std(T.predictionEntropy));
    
    fprintf(fid, '\n═══════════════════════════════════════════════════════════\n');
    fprintf(fid, 'NOTE: These are PATIENT-LEVEL metrics based on majority\n');
    fprintf(fid, '      voting across individual spectrum predictions.\n');
    fprintf(fid, '      NO AVERAGING of spectra was performed.\n');
    fprintf(fid, '═══════════════════════════════════════════════════════════\n');
    
    fclose(fid);
    fprintf('✓ Summary saved to: %s\n', summaryFile);
end
