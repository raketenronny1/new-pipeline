%% METRICS, DISPLAY, AND AGGREGATION FUNCTIONS
% Functions for computing dual-level metrics (spectrum and patient),
% displaying results, and aggregating across folds.
%
% Following best practices from:
% - Baker et al. (2014) Nature Protocols 9(8):1771-1791
% - Greener et al. (2022) Nature Reviews Molecular Cell Biology 23:40-55

function [results] = computeMetrics(results)
    % Computes both spectrum-level and patient-level metrics
    %
    % INPUT/OUTPUT:
    %   results: struct with spectrumLevelResults and patientLevelResults
    
    %% SPECTRUM-LEVEL METRICS (Supplementary)
    specResults = results.spectrumLevelResults;
    results.spectrumMetrics = struct();
    
    y_true_spec = specResults.y_true;
    y_pred_spec = specResults.y_pred;
    
    results.spectrumMetrics.accuracy = mean(y_true_spec == y_pred_spec);
    results.spectrumMetrics.confusionMatrix = confusionmat(y_true_spec, y_pred_spec);
    
    % Calculate sensitivity, specificity, PPV, NPV (spectrum-level)
    CM_spec = results.spectrumMetrics.confusionMatrix;
    
    % Map labels to matrix indices
    unique_labels = unique(y_true_spec);
    if length(unique_labels) == 2
        % Find which label is which
        label1 = unique_labels(1);
        label2 = unique_labels(2);
        
        % Assume lower label is "negative" (WHO-1), higher is "positive" (WHO-3)
        if label1 < label2
            idx_neg = 1; idx_pos = 2;
        else
            idx_neg = 2; idx_pos = 1;
        end
        
        TN = CM_spec(idx_neg, idx_neg);
        FP = CM_spec(idx_neg, idx_pos);
        FN = CM_spec(idx_pos, idx_neg);
        TP = CM_spec(idx_pos, idx_pos);
        
        results.spectrumMetrics.sensitivity = TP / (TP + FN);
        results.spectrumMetrics.specificity = TN / (TN + FP);
        results.spectrumMetrics.PPV = TP / (TP + FP);
        results.spectrumMetrics.NPV = TN / (TN + FN);
        results.spectrumMetrics.F1Score = 2*TP / (2*TP + FP + FN);
    else
        warning('Spectrum-level confusion matrix does not have 2 classes');
    end
    
    %% PATIENT-LEVEL METRICS (Primary)
    patResults = results.patientLevelResults;
    results.patientMetrics = struct();
    
    trueLabels = [patResults.trueLabel];
    predLabels = [patResults.predictedLabel];
    
    results.patientMetrics.accuracy = mean(trueLabels == predLabels);
    results.patientMetrics.confusionMatrix = confusionmat(trueLabels, predLabels);
    
    % Calculate sensitivity, specificity, PPV, NPV (patient-level) - PRIMARY
    CM_pat = results.patientMetrics.confusionMatrix;
    
    % Map labels to matrix indices
    unique_labels_pat = unique(trueLabels);
    if length(unique_labels_pat) == 2
        % Find which label is which
        label1 = unique_labels_pat(1);
        label2 = unique_labels_pat(2);
        
        % Assume lower label is "negative" (WHO-1=1), higher is "positive" (WHO-3=3)
        if label1 < label2
            idx_neg = 1; idx_pos = 2;
        else
            idx_neg = 2; idx_pos = 1;
        end
        
        TN = CM_pat(idx_neg, idx_neg);
        FP = CM_pat(idx_neg, idx_pos);
        FN = CM_pat(idx_pos, idx_neg);
        TP = CM_pat(idx_pos, idx_pos);
        
        results.patientMetrics.sensitivity = TP / (TP + FN);
        results.patientMetrics.specificity = TN / (TN + FP);
        results.patientMetrics.PPV = TP / (TP + FP);
        results.patientMetrics.NPV = TN / (TN + FN);
        results.patientMetrics.F1Score = 2*TP / (2*TP + FP + FN);
        results.patientMetrics.TN = TN;
        results.patientMetrics.FP = FP;
        results.patientMetrics.FN = FN;
        results.patientMetrics.TP = TP;
    else
        warning('Patient-level confusion matrix does not have 2 classes');
    end
    
    %% CONFIDENCE METRICS
    results.confidenceMetrics = struct();
    results.confidenceMetrics.meanConfidence = mean([patResults.majorityVoteConfidence]);
    results.confidenceMetrics.stdConfidence = std([patResults.majorityVoteConfidence]);
    results.confidenceMetrics.meanEntropy = mean([patResults.predictionEntropy]);
    results.confidenceMetrics.stdEntropy = std([patResults.predictionEntropy]);
    
    % Identify high/low confidence predictions
    highConfThresh = 0.85; % >85% spectra agree
    lowConfThresh = 0.60;  % <60% spectra agree
    
    highConfIdx = [patResults.majorityVoteConfidence] >= highConfThresh;
    lowConfIdx = [patResults.majorityVoteConfidence] <= lowConfThresh;
    
    results.confidenceMetrics.nHighConfidence = sum(highConfIdx);
    results.confidenceMetrics.nLowConfidence = sum(lowConfIdx);
    
    if sum(highConfIdx) > 0
        results.confidenceMetrics.accuracyHighConf = mean([patResults(highConfIdx).isCorrect]);
    else
        results.confidenceMetrics.accuracyHighConf = NaN;
    end
    
    if sum(lowConfIdx) > 0
        results.confidenceMetrics.accuracyLowConf = mean([patResults(lowConfIdx).isCorrect]);
    else
        results.confidenceMetrics.accuracyLowConf = NaN;
    end
end


function displayFoldResults(results)
    % Displays results for a single fold
    
    fprintf('\n--- FOLD %d RESULTS ---\n', results.fold);
    
    % Spectrum-level (brief)
    fprintf('\n[Spectrum-Level - Supplementary]\n');
    fprintf('  Accuracy: %.2f%% (%d/%d correct)\n', ...
            results.spectrumMetrics.accuracy * 100, ...
            sum(results.spectrumLevelResults.y_true == results.spectrumLevelResults.y_pred), ...
            length(results.spectrumLevelResults.y_true));
    
    if isfield(results.spectrumMetrics, 'sensitivity')
        fprintf('  Sensitivity: %.2f%%, Specificity: %.2f%%\n', ...
                results.spectrumMetrics.sensitivity * 100, ...
                results.spectrumMetrics.specificity * 100);
    end
    
    % Patient-level (detailed) - PRIMARY
    fprintf('\n[Patient-Level - PRIMARY METRICS]\n');
    nPatients = length(results.patientLevelResults);
    nCorrect = sum([results.patientLevelResults.isCorrect]);
    
    fprintf('  Accuracy:    %.2f%% (%d/%d patients)\n', ...
            results.patientMetrics.accuracy * 100, nCorrect, nPatients);
    
    if isfield(results.patientMetrics, 'sensitivity')
        fprintf('  Sensitivity: %.2f%% (%.0f/%d WHO-3 patients)\n', ...
                results.patientMetrics.sensitivity * 100, ...
                results.patientMetrics.TP, ...
                results.patientMetrics.TP + results.patientMetrics.FN);
        fprintf('  Specificity: %.2f%% (%.0f/%d WHO-1 patients)\n', ...
                results.patientMetrics.specificity * 100, ...
                results.patientMetrics.TN, ...
                results.patientMetrics.TN + results.patientMetrics.FP);
        fprintf('  PPV:         %.2f%%\n', results.patientMetrics.PPV * 100);
        fprintf('  NPV:         %.2f%%\n', results.patientMetrics.NPV * 100);
        fprintf('  F1-Score:    %.3f\n', results.patientMetrics.F1Score);
    end
    
    % Confusion Matrix
    fprintf('\n  Confusion Matrix (Patient-Level):\n');
    CM = results.patientMetrics.confusionMatrix;
    fprintf('                Predicted WHO-1   Predicted WHO-3\n');
    
    % Determine label ordering
    trueLabels = [results.patientLevelResults.trueLabel];
    unique_labels = unique(trueLabels);
    if unique_labels(1) == 1
        fprintf('  True WHO-1:   %8d          %8d\n', CM(1,1), CM(1,2));
        fprintf('  True WHO-3:   %8d          %8d\n', CM(2,1), CM(2,2));
    else
        fprintf('  True WHO-1:   %8d          %8d\n', CM(2,2), CM(2,1));
        fprintf('  True WHO-3:   %8d          %8d\n', CM(1,2), CM(1,1));
    end
    
    % Confidence metrics
    fprintf('\n[Confidence Metrics]\n');
    fprintf('  Mean confidence: %.3f ± %.3f\n', ...
            results.confidenceMetrics.meanConfidence, ...
            results.confidenceMetrics.stdConfidence);
    fprintf('  Mean entropy: %.3f ± %.3f\n', ...
            results.confidenceMetrics.meanEntropy, ...
            results.confidenceMetrics.stdEntropy);
    
    if results.confidenceMetrics.nHighConfidence > 0
        fprintf('  High confidence (>85%% agreement): %d patients (Accuracy: %.2f%%)\n', ...
                results.confidenceMetrics.nHighConfidence, ...
                results.confidenceMetrics.accuracyHighConf * 100);
    end
    
    if results.confidenceMetrics.nLowConfidence > 0
        fprintf('  Low confidence (<60%% agreement): %d patients (Accuracy: %.2f%%)\n', ...
                results.confidenceMetrics.nLowConfidence, ...
                results.confidenceMetrics.accuracyLowConf * 100);
    end
    
    fprintf('\n');
end


function [cvResults] = aggregateCVResults(cvResults)
    % Aggregates results across all folds and computes summary statistics
    
    K = length(cvResults);
    
    % Extract patient-level metrics from all folds
    accuracies = zeros(K, 1);
    sensitivities = zeros(K, 1);
    specificities = zeros(K, 1);
    f1scores = zeros(K, 1);
    ppvs = zeros(K, 1);
    npvs = zeros(K, 1);
    
    for k = 1:K
        accuracies(k) = cvResults(k).patientMetrics.accuracy;
        if isfield(cvResults(k).patientMetrics, 'sensitivity')
            sensitivities(k) = cvResults(k).patientMetrics.sensitivity;
            specificities(k) = cvResults(k).patientMetrics.specificity;
            f1scores(k) = cvResults(k).patientMetrics.F1Score;
            ppvs(k) = cvResults(k).patientMetrics.PPV;
            npvs(k) = cvResults(k).patientMetrics.NPV;
        end
    end
    
    % Store aggregated results
    cvResults(1).aggregated = struct();
    cvResults(1).aggregated.meanAccuracy = mean(accuracies);
    cvResults(1).aggregated.stdAccuracy = std(accuracies);
    cvResults(1).aggregated.ciAccuracy = 1.96 * std(accuracies) / sqrt(K);  % 95% CI
    
    cvResults(1).aggregated.meanSensitivity = mean(sensitivities);
    cvResults(1).aggregated.stdSensitivity = std(sensitivities);
    cvResults(1).aggregated.ciSensitivity = 1.96 * std(sensitivities) / sqrt(K);
    
    cvResults(1).aggregated.meanSpecificity = mean(specificities);
    cvResults(1).aggregated.stdSpecificity = std(specificities);
    cvResults(1).aggregated.ciSpecificity = 1.96 * std(specificities) / sqrt(K);
    
    cvResults(1).aggregated.meanF1Score = mean(f1scores);
    cvResults(1).aggregated.stdF1Score = std(f1scores);
    cvResults(1).aggregated.ciF1Score = 1.96 * std(f1scores) / sqrt(K);
    
    cvResults(1).aggregated.meanPPV = mean(ppvs);
    cvResults(1).aggregated.stdPPV = std(ppvs);
    
    cvResults(1).aggregated.meanNPV = mean(npvs);
    cvResults(1).aggregated.stdNPV = std(npvs);
    
    % Display final results
    fprintf('\n');
    fprintf('═════════════════════════════════════════════════════════════\n');
    fprintf('  FINAL CROSS-VALIDATION RESULTS (%d-Fold)\n', K);
    fprintf('═════════════════════════════════════════════════════════════\n');
    fprintf('Patient-Level Performance (Mean ± SD) [95%% CI]:\n');
    fprintf('  Accuracy:    %.2f%% ± %.2f%% [±%.2f%%]\n', ...
            cvResults(1).aggregated.meanAccuracy * 100, ...
            cvResults(1).aggregated.stdAccuracy * 100, ...
            cvResults(1).aggregated.ciAccuracy * 100);
    fprintf('  Sensitivity: %.2f%% ± %.2f%% [±%.2f%%]\n', ...
            cvResults(1).aggregated.meanSensitivity * 100, ...
            cvResults(1).aggregated.stdSensitivity * 100, ...
            cvResults(1).aggregated.ciSensitivity * 100);
    fprintf('  Specificity: %.2f%% ± %.2f%% [±%.2f%%]\n', ...
            cvResults(1).aggregated.meanSpecificity * 100, ...
            cvResults(1).aggregated.stdSpecificity * 100, ...
            cvResults(1).aggregated.ciSpecificity * 100);
    fprintf('  PPV:         %.2f%% ± %.2f%%\n', ...
            cvResults(1).aggregated.meanPPV * 100, ...
            cvResults(1).aggregated.stdPPV * 100);
    fprintf('  NPV:         %.2f%% ± %.2f%%\n', ...
            cvResults(1).aggregated.meanNPV * 100, ...
            cvResults(1).aggregated.stdNPV * 100);
    fprintf('  F1-Score:    %.3f ± %.3f [±%.3f]\n', ...
            cvResults(1).aggregated.meanF1Score, ...
            cvResults(1).aggregated.stdF1Score, ...
            cvResults(1).aggregated.ciF1Score);
    fprintf('═════════════════════════════════════════════════════════════\n');
    fprintf('\nNOTE: These are PATIENT-LEVEL metrics (majority voting)\n');
    fprintf('      based on individual spectrum predictions.\n');
    fprintf('═════════════════════════════════════════════════════════════\n');
end
