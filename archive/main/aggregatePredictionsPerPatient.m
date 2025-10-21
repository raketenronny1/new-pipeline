function [patientPredictions, patientConfidence] = ...
         aggregatePredictionsPerPatient(y_pred_spectra, y_pred_prob, ...
                                        testPatientIDs, patientData, testPatientIdx)
    % Aggregates spectrum-level predictions to patient-level via majority vote
    %
    % INPUT:
    %   y_pred_spectra: [N_test_spectra × 1] predicted labels for each spectrum
    %   y_pred_prob: [N_test_spectra × 2] predicted probabilities [P(WHO-1), P(WHO-3)]
    %   testPatientIDs: cell array of patient IDs for each test spectrum
    %   patientData: original patient data struct
    %   testPatientIdx: indices of test patients
    %
    % OUTPUT:
    %   patientPredictions: struct array with per-patient results
    %   patientConfidence: struct with confidence metrics per patient
    
    uniquePatients = unique(testPatientIDs);
    nPatients = length(uniquePatients);
    
    patientPredictions = struct('patientID', {}, ...
                                'trueLabel', {}, ...
                                'predictedLabel', {}, ...
                                'nSpectra', {}, ...
                                'nPredictedWHO1', {}, ...
                                'nPredictedWHO3', {}, ...
                                'majorityVoteConfidence', {}, ...
                                'meanProbWHO1', {}, ...
                                'meanProbWHO3', {}, ...
                                'stdProbWHO1', {}, ...
                                'stdProbWHO3', {}, ...
                                'predictionEntropy', {}, ...
                                'isCorrect', {});
    
    for i = 1:nPatients
        patID = uniquePatients{i};
        
        % Find all spectra from this patient
        spectraIdx = strcmp(testPatientIDs, patID);
        
        % Get predictions for this patient's spectra
        patientSpectraPred = y_pred_spectra(spectraIdx);
        patientSpectraProb = y_pred_prob(spectraIdx, :);
        
        % Find true label
        patIdx = testPatientIdx(i);
        trueLabel = patientData(patIdx).label;
        
        % Majority vote
        nWHO1 = sum(patientSpectraPred == 1);
        nWHO3 = sum(patientSpectraPred == 3);
        nTotal = length(patientSpectraPred);
        
        if nWHO1 > nWHO3
            predictedLabel = 1;
            majorityConfidence = nWHO1 / nTotal;
        elseif nWHO3 > nWHO1
            predictedLabel = 3;
            majorityConfidence = nWHO3 / nTotal;
        else
            % Tie: use mean probability
            meanProbWHO3 = mean(patientSpectraProb(:, 2));
            if meanProbWHO3 > 0.5
                predictedLabel = 3;
            else
                predictedLabel = 1;
            end
            majorityConfidence = 0.5; % Indicate uncertainty
        end
        
        % Compute confidence metrics
        meanProb = mean(patientSpectraProb, 1);
        stdProb = std(patientSpectraProb, 0, 1);
        
        % Prediction entropy (uncertainty measure)
        % Add small epsilon to avoid log(0)
        entropy = -sum(meanProb .* log2(meanProb + eps));
        
        % Store results
        patientPredictions(i).patientID = patID;
        patientPredictions(i).trueLabel = trueLabel;
        patientPredictions(i).predictedLabel = predictedLabel;
        patientPredictions(i).nSpectra = nTotal;
        patientPredictions(i).nPredictedWHO1 = nWHO1;
        patientPredictions(i).nPredictedWHO3 = nWHO3;
        patientPredictions(i).majorityVoteConfidence = majorityConfidence;
        patientPredictions(i).meanProbWHO1 = meanProb(1);
        patientPredictions(i).meanProbWHO3 = meanProb(2);
        patientPredictions(i).stdProbWHO1 = stdProb(1);
        patientPredictions(i).stdProbWHO3 = stdProb(2);
        patientPredictions(i).predictionEntropy = entropy;
        patientPredictions(i).isCorrect = (predictedLabel == trueLabel);
    end
    
    patientConfidence = patientPredictions;
end
