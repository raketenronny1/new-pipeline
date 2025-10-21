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
