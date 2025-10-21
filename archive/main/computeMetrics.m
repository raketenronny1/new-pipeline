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
