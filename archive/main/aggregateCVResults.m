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
