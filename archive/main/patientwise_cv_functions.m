%% PATIENT-WISE STRATIFIED CROSS-VALIDATION FUNCTIONS
% Implementation of patient-wise stratified K-fold cross-validation
% to prevent data leakage across folds.
%
% Following best practices from:
% - Baker et al. (2014) Nature Protocols 9(8):1771-1791
% - Greener et al. (2022) Nature Reviews Molecular Cell Biology 23:40-55
%
% Key principles:
% - All spectra from one patient stay together in the same fold
% - Stratified sampling maintains class balance across folds
% - No data leakage between train/test splits

function [cvFolds] = createPatientWiseStratifiedCV(patientData, K, random_seed)
    % Creates K-fold cross-validation splits ensuring:
    % 1. Each patient's ALL spectra stay together in one fold
    % 2. Class balance is maintained across folds (stratified)
    % 3. No data leakage between folds
    %
    % INPUT:
    %   patientData: struct array with fields patientID, spectra, label
    %   K: number of folds (recommend K=5 or K=10)
    %   random_seed: (optional) random seed for reproducibility
    %
    % OUTPUT:
    %   cvFolds: struct array with K elements
    %            cvFolds(k).trainPatientIdx = [indices of training patients]
    %            cvFolds(k).testPatientIdx = [indices of test patients]
    
    if nargin < 3
        random_seed = 42;
    end
    
    nPatients = length(patientData);
    labels = [patientData.label];
    
    fprintf('\n=== Creating Patient-Wise Stratified %d-Fold CV ===\n', K);
    fprintf('Total patients: %d\n', nPatients);
    
    % Separate patient indices by class
    idxWHO1 = find(labels == 1);
    idxWHO3 = find(labels == 3);
    
    fprintf('WHO-1 patients: %d\n', length(idxWHO1));
    fprintf('WHO-3 patients: %d\n', length(idxWHO3));
    
    % Shuffle within each class (for randomization)
    rng(random_seed, 'twister');
    idxWHO1 = idxWHO1(randperm(length(idxWHO1)));
    idxWHO3 = idxWHO3(randperm(length(idxWHO3)));
    
    % Create K folds for each class
    foldsWHO1 = createFolds(idxWHO1, K);
    foldsWHO3 = createFolds(idxWHO3, K);
    
    % Combine folds
    cvFolds = struct('trainPatientIdx', {}, 'testPatientIdx', {});
    
    fprintf('\nFold composition:\n');
    fprintf('%-6s | %-20s | %-20s\n', 'Fold', 'Train (WHO-1/WHO-3)', 'Test (WHO-1/WHO-3)');
    fprintf('%s\n', repmat('-', 1, 55));
    
    for k = 1:K
        testIdx = [foldsWHO1{k}; foldsWHO3{k}];
        trainIdx = setdiff(1:nPatients, testIdx);
        
        cvFolds(k).trainPatientIdx = trainIdx;
        cvFolds(k).testPatientIdx = testIdx;
        
        % Validation: check no overlap
        assert(isempty(intersect(trainIdx, testIdx)), ...
               sprintf('Fold %d has overlapping train/test patients!', k));
        
        % Report fold composition
        trainLabels = labels(trainIdx);
        testLabels = labels(testIdx);
        
        fprintf('%-6d | %3d / %3d  (n=%2d)    | %2d / %2d  (n=%d)\n', ...
                k, ...
                sum(trainLabels==1), sum(trainLabels==3), length(trainIdx), ...
                sum(testLabels==1), sum(testLabels==3), length(testIdx));
        
        % Additional validation: ensure both classes present in training
        if sum(trainLabels==1) == 0 || sum(trainLabels==3) == 0
            error('Fold %d has only one class in training set!', k);
        end
    end
    
    fprintf('%s\n', repmat('-', 1, 55));
    fprintf('✓ Cross-validation folds created successfully\n');
    fprintf('✓ No patient appears in both train and test within any fold\n');
end


function folds = createFolds(indices, K)
    % Helper function to split indices into K approximately equal folds
    %
    % INPUT:
    %   indices: array of patient indices to split
    %   K: number of folds
    %
    % OUTPUT:
    %   folds: cell array of K folds
    
    n = length(indices);
    foldSize = floor(n / K);
    folds = cell(K, 1);
    
    for k = 1:K-1
        startIdx = (k-1)*foldSize + 1;
        endIdx = k*foldSize;
        folds{k} = indices(startIdx:endIdx);
    end
    % Last fold gets remaining indices
    folds{K} = indices((K-1)*foldSize+1:end);
end


function [X_train, y_train, X_test, y_test, testPatientIDs, testSpectrumToPatientMap] = ...
         extractSpectraForFold(patientData, trainPatientIdx, testPatientIdx)
    % Extracts all spectra from train/test patients and creates labels
    %
    % INPUT:
    %   patientData: struct array with patient data
    %   trainPatientIdx: indices of patients for training
    %   testPatientIdx: indices of patients for testing
    %
    % OUTPUT:
    %   X_train: [N_train_spectra × N_wavenumbers] training spectra
    %   y_train: [N_train_spectra × 1] labels (1 or 3) for each spectrum
    %   X_test:  [N_test_spectra × N_wavenumbers] test spectra
    %   y_test:  [N_test_spectra × 1] labels for each spectrum
    %   testPatientIDs: cell array mapping test spectrum to patient
    %   testSpectrumToPatientMap: struct mapping spectrum index to patient info
    
    % Training data
    X_train = [];
    y_train = [];
    
    for i = 1:length(trainPatientIdx)
        patIdx = trainPatientIdx(i);
        spectra = patientData(patIdx).spectra;  % [N_spectra × N_wn]
        label = patientData(patIdx).label;
        
        X_train = [X_train; spectra];
        y_train = [y_train; repmat(label, size(spectra, 1), 1)];
    end
    
    % Test data
    X_test = [];
    y_test = [];
    testPatientIDs = {};
    testSpectrumToPatientMap = struct('spectrumIdx', {}, 'patientIdx', {}, 'patientID', {}, 'localSpectrumIdx', {});
    
    spectrum_counter = 0;
    for i = 1:length(testPatientIdx)
        patIdx = testPatientIdx(i);
        spectra = patientData(patIdx).spectra;
        label = patientData(patIdx).label;
        patID = patientData(patIdx).patientID;
        
        nSpectra = size(spectra, 1);
        X_test = [X_test; spectra];
        y_test = [y_test; repmat(label, nSpectra, 1)];
        testPatientIDs = [testPatientIDs; repmat({patID}, nSpectra, 1)];
        
        % Create detailed mapping
        for j = 1:nSpectra
            spectrum_counter = spectrum_counter + 1;
            testSpectrumToPatientMap(spectrum_counter).spectrumIdx = spectrum_counter;
            testSpectrumToPatientMap(spectrum_counter).patientIdx = patIdx;
            testSpectrumToPatientMap(spectrum_counter).patientID = patID;
            testSpectrumToPatientMap(spectrum_counter).localSpectrumIdx = j;
        end
    end
    
    % Report extraction summary
    fprintf('  Extracted spectra: Train=%d, Test=%d\n', size(X_train,1), size(X_test,1));
    
    % Validate no NaN/Inf
    if any(isnan(X_train(:))) || any(isinf(X_train(:)))
        error('Training spectra contain NaN or Inf values!');
    end
    if any(isnan(X_test(:))) || any(isinf(X_test(:)))
        error('Test spectra contain NaN or Inf values!');
    end
end


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
