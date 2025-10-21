%% VISUALIZATION AND EXPORT FUNCTIONS
% Functions for creating visualizations and exporting results for clinical review
%
% Following best practices from:
% - Baker et al. (2014) Nature Protocols 9(8):1771-1791
% - Greener et al. (2022) Nature Reviews Molecular Cell Biology 23:40-55

function visualizePatientConfidence(cvResults, output_dir)
    % Creates comprehensive visualization of patient-level predictions
    %
    % INPUT:
    %   cvResults: struct array with CV results
    %   output_dir: directory to save figures
    
    if nargin < 2
        output_dir = 'results/meningioma_ftir_pipeline/';
    end
    
    % Aggregate all patient predictions across folds
    allPatientResults = [];
    for k = 1:length(cvResults)
        if isfield(cvResults(k), 'patientLevelResults')
            allPatientResults = [allPatientResults; cvResults(k).patientLevelResults];
        end
    end
    
    if isempty(allPatientResults)
        warning('No patient results to visualize');
        return;
    end
    
    % Create figure
    figure('Position', [100, 100, 1400, 900], 'Color', 'white');
    
    % Extract data
    confidences = [allPatientResults.majorityVoteConfidence];
    isCorrect = [allPatientResults.isCorrect];
    entropies = [allPatientResults.predictionEntropy];
    meanProbWHO3 = [allPatientResults.meanProbWHO3];
    stdProbWHO3 = [allPatientResults.stdProbWHO3];
    nSpectra = [allPatientResults.nSpectra];
    nPredWHO1 = [allPatientResults.nPredictedWHO1];
    nPredWHO3 = [allPatientResults.nPredictedWHO3];
    trueLabels = [allPatientResults.trueLabel];
    predLabels = [allPatientResults.predictedLabel];
    
    %% Subplot 1: Histogram of majority vote confidence
    subplot(2, 3, 1);
    histogram(confidences, 20, 'FaceColor', [0.3 0.6 0.9]);
    xlabel('Majority Vote Confidence', 'FontSize', 11);
    ylabel('Number of Patients', 'FontSize', 11);
    title('Distribution of Prediction Confidence', 'FontSize', 12, 'FontWeight', 'bold');
    xline(0.85, 'r--', 'LineWidth', 2, 'Label', 'High Conf');
    xline(0.60, 'r--', 'LineWidth', 2, 'Label', 'Low Conf');
    grid on;
    
    %% Subplot 2: Confidence vs Correctness
    subplot(2, 3, 2);
    boxplot(confidences, isCorrect, 'Labels', {'Incorrect', 'Correct'}, ...
            'Colors', [0.8 0.3 0.3; 0.3 0.8 0.3]);
    ylabel('Majority Vote Confidence', 'FontSize', 11);
    title('Confidence by Prediction Correctness', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    
    %% Subplot 3: Entropy distribution
    subplot(2, 3, 3);
    histogram(entropies, 20, 'FaceColor', [0.9 0.6 0.3]);
    xlabel('Prediction Entropy', 'FontSize', 11);
    ylabel('Number of Patients', 'FontSize', 11);
    title('Uncertainty Distribution', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    
    %% Subplot 4: Scatter plot - mean prob vs std
    subplot(2, 3, 4);
    scatter(meanProbWHO3, stdProbWHO3, 80, confidences, 'filled', 'MarkerEdgeColor', 'k');
    xlabel('Mean P(WHO-3)', 'FontSize', 11);
    ylabel('Std P(WHO-3)', 'FontSize', 11);
    title('Prediction Mean vs Variability', 'FontSize', 12, 'FontWeight', 'bold');
    cb = colorbar;
    cb.Label.String = 'Confidence';
    colormap('jet');
    grid on;
    
    %% Subplot 5: Agreement across spectra
    subplot(2, 3, 5);
    nAgreed = max(nPredWHO1, nPredWHO3);
    agreementPct = (nAgreed ./ nSpectra) * 100;
    histogram(agreementPct, 20, 'FaceColor', [0.6 0.3 0.9]);
    xlabel('% Spectra Agreeing on Majority Class', 'FontSize', 11);
    ylabel('Number of Patients', 'FontSize', 11);
    title('Spectrum Agreement Distribution', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    
    %% Subplot 6: Confusion matrix heatmap
    subplot(2, 3, 6);
    CM = confusionmat(trueLabels, predLabels);
    
    imagesc(CM);
    colorbar;
    colormap(gca, 'hot');
    
    % Determine label ordering
    unique_labels = unique(trueLabels);
    if unique_labels(1) == 1
        xticks([1 2]); xticklabels({'WHO-1', 'WHO-3'});
        yticks([1 2]); yticklabels({'WHO-1', 'WHO-3'});
    else
        xticks([1 2]); xticklabels({'WHO-3', 'WHO-1'});
        yticks([1 2]); yticklabels({'WHO-3', 'WHO-1'});
    end
    
    xlabel('Predicted', 'FontSize', 11);
    ylabel('True', 'FontSize', 11);
    title('Patient-Level Confusion Matrix', 'FontSize', 12, 'FontWeight', 'bold');
    
    % Add text annotations
    for i = 1:size(CM,1)
        for j = 1:size(CM,2)
            text(j, i, num2str(CM(i,j)), 'HorizontalAlignment', 'center', ...
                 'Color', 'white', 'FontSize', 18, 'FontWeight', 'bold');
        end
    end
    
    % Overall title
    sgtitle('Patient-Level Classification Analysis', 'FontSize', 16, 'FontWeight', 'bold');
    
    % Save figure
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    saveas(gcf, fullfile(output_dir, 'patient_confidence_analysis.png'));
    saveas(gcf, fullfile(output_dir, 'patient_confidence_analysis.fig'));
    
    fprintf('✓ Visualization saved to: %s\n', output_dir);
end


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
    fprintf('✓ Summary statistics saved to: %s\n', summaryFile);
end
