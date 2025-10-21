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
    foldsWHO1 = createFolds_local(idxWHO1, K);
    foldsWHO3 = createFolds_local(idxWHO3, K);
    
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


function folds = createFolds_local(indices, K)
    % Helper function to split indices into K approximately equal folds
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
