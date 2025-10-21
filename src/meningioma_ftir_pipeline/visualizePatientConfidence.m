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
