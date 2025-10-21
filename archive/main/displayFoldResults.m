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
