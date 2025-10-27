% Debug EDA issue
addpath('src/meningioma_ftir_pipeline');

fprintf('=== DEBUGGING EDA ISSUE ===\n\n');

% Load dataset
fprintf('1. Loading complete dataset...\n');
load('data/dataset_complete.mat', 'dataset_men');
fprintf('   Dataset size: %d probes\n', height(dataset_men));
fprintf('   Columns: %s\n', strjoin(dataset_men.Properties.VariableNames, ', '));

% Check ProbeUID
if ismember('ProbeUID', dataset_men.Properties.VariableNames)
    fprintf('   ProbeUID range: %d to %d\n', min(dataset_men.ProbeUID), max(dataset_men.ProbeUID));
    fprintf('   ProbeUID type: %s\n', class(dataset_men.ProbeUID));
else
    fprintf('   ERROR: ProbeUID not found!\n');
end

% Load train data
fprintf('\n2. Loading train data...\n');
load('data/data_table_train.mat', 'data_table_train');
fprintf('   Train size: %d probes\n', height(data_table_train));
fprintf('   Train columns: %s\n', strjoin(data_table_train.Properties.VariableNames, ', '));

if ismember('ProbeUID', data_table_train.Properties.VariableNames)
    fprintf('   Train ProbeUID range: %d to %d\n', min(data_table_train.ProbeUID), max(data_table_train.ProbeUID));
    fprintf('   Train ProbeUID type: %s\n', class(data_table_train.ProbeUID));
    
    % Try to create train_indices
    fprintf('\n3. Creating train_indices...\n');
    train_probe_uids = data_table_train.ProbeUID;
    train_indices = ismember(dataset_men.ProbeUID, train_probe_uids);
    fprintf('   Train indices sum: %d\n', sum(train_indices));
    fprintf('   Train indices type: %s\n', class(train_indices));
    
    % Try to extract spectra
    fprintf('\n4. Testing spectrum extraction...\n');
    try
        test_spectra = dataset_men.CombinedSpectra_PP1{1};
        fprintf('   First sample spectra: %d × %d\n', size(test_spectra));
    catch ME
        fprintf('   ERROR extracting spectra: %s\n', ME.message);
    end
    
    % Try to build all_spectra like EDA does
    fprintf('\n5. Testing EDA data preparation...\n');
    try
        all_spectra = [];
        all_is_train = [];
        for i = 1:min(3, height(dataset_men))
            spectra_matrix = dataset_men.CombinedSpectra_PP1{i};
            n_spectra_probe = size(spectra_matrix, 1);
            
            all_spectra = [all_spectra; spectra_matrix];
            is_train_probe = train_indices(i);
            all_is_train = [all_is_train; repmat(is_train_probe, n_spectra_probe, 1)];
            
            fprintf('   Probe %d: %d spectra, train=%d\n', i, n_spectra_probe, is_train_probe);
        end
        fprintf('   Total spectra accumulated: %d\n', size(all_spectra, 1));
        fprintf('   Train flags: %d\n', sum(all_is_train));
    catch ME
        fprintf('   ERROR in data preparation: %s\n', ME.message);
        for j = 1:length(ME.stack)
            fprintf('     at %s (line %d)\n', ME.stack(j).name, ME.stack(j).line);
        end
    end
    
else
    fprintf('   ERROR: ProbeUID not found in train data!\n');
end

fprintf('\n=== DEBUG COMPLETE ===\n');
