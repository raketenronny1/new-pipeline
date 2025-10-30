function dataset_men = prepare_ftir_dataset(varargin)
% PREPARE_FTIR_DATASET Create complete FTIR dataset with dual preprocessing
%
% SYNTAX:
%   dataset_men = prepare_ftir_dataset()
%   dataset_men = prepare_ftir_dataset('DataDir', path)
%   dataset_men = prepare_ftir_dataset('SaveOutput', true)
%
% INPUTS (Name-Value Pairs):
%   'DataDir'     - Path to data directory (default: '../../data')
%   'SaveOutput'  - Save dataset_complete.mat (default: true)
%   'Verbose'     - Display progress messages (default: true)
%
% OUTPUTS:
%   dataset_men   - Complete probe-level dataset table with columns:
%                   ProbeUID, Diss_ID, Patient_ID, Fall_ID, Age, NumPositions,
%                   PositionSpectra, NumTotalSpectra, CombinedRawSpectra,
%                   CombinedSpectra_PP1, CombinedSpectra_PP2, MeanSpectrum_PP1,
%                   MeanSpectrum_PP2, WHO_Grade, Sex, Subtyp, methylation_class,
%                   methylation_cluster
%
% DESCRIPTION:
%   Recreates the complete FTIR dataset from raw data with enhanced preprocessing.
%   
%   Processing steps:
%   0.5. Deduplicate measurements (NEW - matches old dataset)
%   1. Load allspekTable.mat and metadata_all_patients.mat
%   2. Join tables by Proben_ID
%   3. Apply dual preprocessing (PP1 and PP2) at position level
%   4. Aggregate positions into probe-level data
%   5. Format categorical variables
%   6. Calculate representative spectra (mean)
%   7. Save as dataset_complete.mat
%
%   Preprocessing Approaches:
%   - PP1 (Standard): Vector norm + 2nd derivative
%   - PP2 (Enhanced): Bin(4) + Smooth + Vector norm + 2nd derivative
%
%   DEDUPLICATION (Step 0.5):
%   Some samples have duplicate measurements from different measurement sessions.
%   Strategy: Keep the session with LOWEST session number (earliest measurement).
%   This matches the old dataset and ensures consistency with validated results.
%   
%   Known duplicates (as of 2025-10-24):
%   - DD004-T001 (MEN-080-01): S11 and S25 → Keep S11 (r=0.998 with old)
%   - DD007-T001 (MEN-083-01): S4 and S18 → Keep S4 (r=0.992 with old)
%
% EXAMPLE:
%   % Create dataset with default settings
%   dataset_men = prepare_ftir_dataset();
%   
%   % Create without saving
%   dataset_men = prepare_ftir_dataset('SaveOutput', false);
%
% See also: PREPROCESS_SPECTRA, CREATE_PREPROCESSING_CONFIG
%
% Author: GitHub Copilot
% Date: 2025-10-24

%% Parse input arguments
p = inputParser;
addParameter(p, 'DataDir', fullfile(fileparts(mfilename('fullpath')), '..', '..', 'data'), @ischar);
addParameter(p, 'SaveOutput', true, @islogical);
addParameter(p, 'Verbose', true, @islogical);
parse(p, varargin{:});

data_dir = p.Results.DataDir;
save_output = p.Results.SaveOutput;
verbose = p.Results.Verbose;

%% Add preprocessing functions to path
preprocessing_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'preprocessing');
addpath(preprocessing_dir);

%% Display header
if verbose
    fprintf('\n');
    fprintf('========================================================================\n');
    fprintf('  FTIR Dataset Preparation Pipeline\n');
    fprintf('  Dual Preprocessing: PP1 (Standard) + PP2 (Enhanced)\n');
    fprintf('========================================================================\n\n');
end

%% Step 0: Load prerequisite data
if verbose
    fprintf('Step 0: Loading prerequisite data...\n');
end

% Load allspekTable
allspek_file = fullfile(data_dir, 'allspekTable.mat');
if ~exist(allspek_file, 'file')
    error('prepare_ftir_dataset:FileNotFound', ...
        'allspekTable.mat not found at: %s', allspek_file);
end
load(allspek_file, 'allspekTable');
if verbose
    fprintf('  Loaded allspekTable: %d positions\n', height(allspekTable));
end

% Load metadata
metadata_file = fullfile(data_dir, 'metadata_all_patients.mat');
if ~exist(metadata_file, 'file')
    error('prepare_ftir_dataset:FileNotFound', ...
        'metadata_all_patients.mat not found at: %s', metadata_file);
end
load(metadata_file, 'metadata_patients');
if verbose
    fprintf('  Loaded metadata_patients: %d patients\n', height(metadata_patients));
end

% Load wavenumbers
wavenumber_file = fullfile(data_dir, 'wavenumbers.mat');
if ~exist(wavenumber_file, 'file')
    error('prepare_ftir_dataset:FileNotFound', ...
        'wavenumbers.mat not found at: %s', wavenumber_file);
end
load(wavenumber_file, 'wavenumbers_roi');
if iscolumn(wavenumbers_roi)
    wavenumbers_roi = wavenumbers_roi';
end
if verbose
    fprintf('  Loaded wavenumbers: %d points (%.1f-%.1f cm^-1)\n', ...
        length(wavenumbers_roi), max(wavenumbers_roi), min(wavenumbers_roi));
end

%% Step 0.5: Deduplicate allspekTable
% CRITICAL: Some samples have duplicate measurements from different sessions
% Example: DD004-T001 has both S11 and S25 sessions (6 positions total instead of 3)
%          DD007-T001 has both S4 and S18 sessions (6 positions total instead of 3)
%
% DEDUPLICATION STRATEGY (to match old dataset):
% - For DD004-T001 (MEN-080-01): Keep S11 session (correlation r=0.998 with old dataset)
% - For DD007-T001 (MEN-083-01): Keep S4 session (correlation r=0.992 with old dataset)
% - Rule: When multiple sessions exist, keep the one with LOWEST session number (earliest)
%
% Quality Analysis (2025-10-24):
%   MEN-080-01: S11 has r=0.998 vs S25 r=0.762 correlation with old dataset
%   MEN-083-01: S4 has r=0.992 vs S18 r=0.945 correlation with old dataset
%
% This ensures consistency with previous validated analysis results.

if verbose
    fprintf('\nStep 0.5: Deduplicating measurements...\n');
end

% Identify duplicate Proben_ID + Position combinations
[unique_combos, ~, combo_idx] = unique(strcat(allspekTable.Proben_ID_str, '_', allspekTable.Position), 'stable');
n_unique = length(unique_combos);
n_original = height(allspekTable);

if n_unique < n_original
    if verbose
        fprintf('  Found %d duplicate measurements (total: %d, unique: %d)\n', ...
            n_original - n_unique, n_original, n_unique);
    end
    
    % For each unique combo, keep only the first occurrence (lowest session number)
    % This matches the old dataset selection criteria
    keep_indices = false(height(allspekTable), 1);
    for i = 1:n_unique
        matching_rows = find(combo_idx == i);
        
        if length(matching_rows) > 1
            % Multiple measurements exist - extract session numbers from SourceFile
            session_numbers = zeros(length(matching_rows), 1);
            for j = 1:length(matching_rows)
                source_file = allspekTable.SourceFile{matching_rows(j)};
                % Extract session number (e.g., "S11" from "DD004-T001-S11_Pos1.0.mat")
                session_match = regexp(source_file, '-S(\d+)', 'tokens');
                if ~isempty(session_match)
                    session_numbers(j) = str2double(session_match{1}{1});
                else
                    session_numbers(j) = inf; % If no session number, treat as last priority
                end
            end
            
            % Keep the measurement with the LOWEST session number
            [~, min_idx] = min(session_numbers);
            keep_idx = matching_rows(min_idx);
            
            if verbose && length(matching_rows) > 1
                fprintf('  %s: Found %d sessions, keeping session S%d\n', ...
                    unique_combos{i}, length(matching_rows), session_numbers(min_idx));
            end
        else
            % Only one measurement - keep it
            keep_idx = matching_rows(1);
        end
        
        keep_indices(keep_idx) = true;
    end
    
    % Apply deduplication
    allspekTable = allspekTable(keep_indices, :);
    
    if verbose
        fprintf('  After deduplication: %d positions\n', height(allspekTable));
        fprintf('  Removed: %d duplicate measurements\n', n_original - height(allspekTable));
    end
else
    if verbose
        fprintf('  No duplicates found\n');
    end
end

%% Step 1: Join allspekTable with metadata
if verbose
    fprintf('\nStep 1: Joining allspekTable with metadata...\n');
end

% Prepare join keys
allspekTable_j = allspekTable;
allspekTable_j.Proben_ID_join_key = string(strtrim(cellstr(allspekTable_j.Proben_ID_str)));

metadata_patients_j = metadata_patients;
if isnumeric(metadata_patients_j.Proben_ID)
    metadata_patients_j.Proben_ID_join_key = string(metadata_patients_j.Proben_ID);
else
    metadata_patients_j.Proben_ID_join_key = string(strtrim(cellstr(metadata_patients_j.Proben_ID)));
end

% Perform inner join
data_all_positions = innerjoin(allspekTable_j, metadata_patients_j, ...
    'LeftKeys', 'Proben_ID_join_key', ...
    'RightKeys', 'Proben_ID_join_key', ...
    'RightVariables', setdiff(metadata_patients_j.Properties.VariableNames, ...
                             {'Proben_ID_join_key', 'Proben_ID'}));

% Remove join key
if ismember('Proben_ID_join_key', data_all_positions.Properties.VariableNames)
    data_all_positions.Proben_ID_join_key = [];
end

if verbose
    fprintf('  Joined table created: %d position records\n', height(data_all_positions));
    if height(data_all_positions) == 0
        error('prepare_ftir_dataset:NoMatches', 'No matches found in join.');
    end
end

%% Step 2: Apply dual preprocessing at position level
if verbose
    fprintf('\nStep 2: Applying dual preprocessing to position-level data...\n');
    fprintf('  This may take several minutes...\n');
end

num_positions = height(data_all_positions);
num_wavenumbers = length(wavenumbers_roi);

% Initialize storage for preprocessed spectra
ProcessedSpectra_PP1 = cell(num_positions, 1);
ProcessedSpectra_PP2 = cell(num_positions, 1);
ProcessedWavenumbers_PP2 = cell(num_positions, 1);

% Create preprocessing configs
cfg_pp1 = create_preprocessing_config('PP1');
cfg_pp2 = create_preprocessing_config('PP2');

% Progress tracking
if verbose
    fprintf('  Processing %d positions:\n', num_positions);
    progress_interval = max(1, floor(num_positions / 10));
end

tic;
for i = 1:num_positions
    raw_block = data_all_positions.RawSpectrum{i};
    
    % Validate raw spectrum
    if isempty(raw_block) || ~ismatrix(raw_block) || size(raw_block, 2) ~= num_wavenumbers
        warning('prepare_ftir_dataset:InvalidSpectrum', ...
            'Position %d (Diss_ID %s, Pos %s): Invalid RawSpectrum. Skipping preprocessing.', ...
            i, string(data_all_positions.Diss_ID(i)), string(data_all_positions.Position{i}));
        ProcessedSpectra_PP1{i} = raw_block;
        ProcessedSpectra_PP2{i} = raw_block;
        ProcessedWavenumbers_PP2{i} = wavenumbers_roi;
        continue;
    end
    
    % Apply PP1 preprocessing (suppress console output)
    evalc('[pp1_spectra, ~] = preprocess_spectra(raw_block, wavenumbers_roi, cfg_pp1);');
    ProcessedSpectra_PP1{i} = pp1_spectra;
    
    % Apply PP2 preprocessing (suppress console output)
    evalc('[pp2_spectra, pp2_wn] = preprocess_spectra(raw_block, wavenumbers_roi, cfg_pp2);');
    ProcessedSpectra_PP2{i} = pp2_spectra;
    ProcessedWavenumbers_PP2{i} = pp2_wn;
    
    % Progress update
    if verbose && (i == 1 || mod(i, progress_interval) == 0 || i == num_positions)
        elapsed = toc;
        rate = i / elapsed;
        remaining = (num_positions - i) / rate;
        fprintf('    [%d/%d] %.1f%% complete (%.1f pos/s, ~%.0f s remaining)\n', ...
            i, num_positions, 100*i/num_positions, rate, remaining);
    end
end

preprocessing_time = toc;
if verbose
    fprintf('  Preprocessing complete: %.1f seconds (%.2f pos/s)\n\n', ...
        preprocessing_time, num_positions / preprocessing_time);
end

% Add preprocessed spectra to table
data_all_positions.ProcessedSpectra_PP1 = ProcessedSpectra_PP1;
data_all_positions.ProcessedSpectra_PP2 = ProcessedSpectra_PP2;
data_all_positions.ProcessedWavenumbers_PP2 = ProcessedWavenumbers_PP2;

%% Step 3: Aggregate to probe level
if verbose
    fprintf('Step 3: Aggregating positions to probe level...\n');
end

unique_diss_ids = unique(data_all_positions.Diss_ID);
num_probes = length(unique_diss_ids);

if verbose
    fprintf('  Found %d unique probes\n', num_probes);
end

% Initialize cell array for table construction
num_cols = 18;  % Updated column count
probe_data_cell = cell(num_probes, num_cols);

for i = 1:num_probes
    current_diss_id = unique_diss_ids{i};
    idx_this_probe = strcmp(data_all_positions.Diss_ID, current_diss_id);
    positions_this_probe = data_all_positions(idx_this_probe, :);
    num_positions_probe = height(positions_this_probe);
    
    % Aggregate raw spectra
    combined_raw = [];
    for k = 1:num_positions_probe
        raw_block = positions_this_probe.RawSpectrum{k};
        if ~isempty(raw_block) && ismatrix(raw_block)
            combined_raw = [combined_raw; raw_block];
        end
    end
    
    % Aggregate PP1 spectra
    combined_pp1 = [];
    for k = 1:num_positions_probe
        pp1_block = positions_this_probe.ProcessedSpectra_PP1{k};
        if ~isempty(pp1_block) && ismatrix(pp1_block)
            combined_pp1 = [combined_pp1; pp1_block];
        end
    end
    
    % Aggregate PP2 spectra
    combined_pp2 = [];
    for k = 1:num_positions_probe
        pp2_block = positions_this_probe.ProcessedSpectra_PP2{k};
        if ~isempty(pp2_block) && ismatrix(pp2_block)
            combined_pp2 = [combined_pp2; pp2_block];
        end
    end
    
    % Get PP2 wavenumbers (should be same for all positions)
    wn_pp2 = positions_this_probe.ProcessedWavenumbers_PP2{1};
    
    % Create PositionSpectra detail (position name + processed spectra)
    position_spectra_detail = cell(num_positions_probe, 2);
    for k = 1:num_positions_probe
        position_spectra_detail{k, 1} = positions_this_probe.Position{k};
        position_spectra_detail{k, 2} = positions_this_probe.ProcessedSpectra_PP1{k};
    end
    
    % Calculate mean spectra
    if ~isempty(combined_pp1)
        mean_pp1 = mean(combined_pp1, 1, 'omitnan');
    else
        mean_pp1 = NaN(1, num_wavenumbers);
    end
    
    if ~isempty(combined_pp2)
        mean_pp2 = mean(combined_pp2, 1, 'omitnan');
    else
        mean_pp2 = NaN(1, length(wn_pp2));
    end
    
    % Get metadata from first position
    meta = positions_this_probe(1, :);
    
    % Extract methylation data (take first element to ensure scalar)
    methylation_class_val = meta.methylation_class(1);
    methylation_cluster_val = meta.methylation_cluster(1);
    
    % Format categorical variables (will be converted later)
    who_grade_str = format_who_grade(meta.WHO_Grade(1));
    sex_str = format_sex(meta.Sex(1));
    subtyp_str = format_subtyp(meta.Subtyp(1));
    
    % Store in cell array - assign each element individually to avoid dimension issues
    probe_data_cell{i, 1} = i;  % ProbeUID
    probe_data_cell{i, 2} = current_diss_id;  % Diss_ID (keep as char, will be in cell column)
    probe_data_cell{i, 3} = string(meta.Patient(1));  % Patient_ID (string)
    probe_data_cell{i, 4} = double(meta.Fall_ID(1));  % Fall_ID
    probe_data_cell{i, 5} = double(meta.Age(1));  % Age
    probe_data_cell{i, 6} = num_positions_probe;  % NumPositions
    probe_data_cell{i, 7} = position_spectra_detail;  % PositionSpectra (cell array)
    probe_data_cell{i, 8} = size(combined_raw, 1);  % NumTotalSpectra
    probe_data_cell{i, 9} = combined_raw;  % CombinedRawSpectra (matrix directly in cell)
    probe_data_cell{i, 10} = combined_pp1;  % CombinedSpectra_PP1 (matrix directly in cell)
    probe_data_cell{i, 11} = combined_pp2;  % CombinedSpectra_PP2 (matrix directly in cell)
    probe_data_cell{i, 12} = mean_pp1;  % MeanSpectrum_PP1 (vector directly in cell)
    probe_data_cell{i, 13} = mean_pp2;  % MeanSpectrum_PP2 (vector directly in cell)
    probe_data_cell{i, 14} = who_grade_str;  % WHO_Grade string (will convert to categorical later)
    probe_data_cell{i, 15} = sex_str;  % Sex string (will convert to categorical later)
    probe_data_cell{i, 16} = subtyp_str;  % Subtyp string (will convert to categorical later)
    
    % Handle categorical variables - preserve undefined as empty string for proper conversion
    if iscategorical(methylation_class_val) && isundefined(methylation_class_val)
        probe_data_cell{i, 17} = '';
    else
        probe_data_cell{i, 17} = char(methylation_class_val);
    end
    
    if iscategorical(methylation_cluster_val) && isundefined(methylation_cluster_val)
        probe_data_cell{i, 18} = '';
    else
        probe_data_cell{i, 18} = char(methylation_cluster_val);
    end
end

% Create table directly (avoid cell2table issues with categorical arrays)
if verbose
    fprintf('  Creating dataset table...\n');
end

% Extract columns from cell array
ProbeUID = [probe_data_cell{:, 1}]';
Diss_ID = probe_data_cell(:, 2);
Patient_ID = [probe_data_cell{:, 3}]';
Fall_ID = [probe_data_cell{:, 4}]';
Age = [probe_data_cell{:, 5}]';
NumPositions = [probe_data_cell{:, 6}]';
PositionSpectra = probe_data_cell(:, 7);
NumTotalSpectra = [probe_data_cell{:, 8}]';
CombinedRawSpectra = probe_data_cell(:, 9);
CombinedSpectra_PP1 = probe_data_cell(:, 10);
CombinedSpectra_PP2 = probe_data_cell(:, 11);
MeanSpectrum_PP1 = probe_data_cell(:, 12);
MeanSpectrum_PP2 = probe_data_cell(:, 13);

% Convert strings to categorical for consistency with original dataset
WHO_Grade = categorical(probe_data_cell(:, 14));
Sex = categorical(probe_data_cell(:, 15));
Subtyp = categorical(probe_data_cell(:, 16));
methylation_class = categorical(probe_data_cell(:, 17));
methylation_cluster = categorical(probe_data_cell(:, 18));

% Build table
dataset_men = table(ProbeUID, Diss_ID, Patient_ID, Fall_ID, Age, NumPositions, ...
    PositionSpectra, NumTotalSpectra, CombinedRawSpectra, ...
    CombinedSpectra_PP1, CombinedSpectra_PP2, ...
    MeanSpectrum_PP1, MeanSpectrum_PP2, ...
    WHO_Grade, Sex, Subtyp, methylation_class, methylation_cluster);

if verbose
    fprintf('  Probe-level table created: %d probes\n', height(dataset_men));
end

%% Step 4: Format categorical variables
%% Step 4: Summary statistics
if verbose
    fprintf('\n========================================================================\n');
    fprintf('  Dataset Preparation Complete!\n');
    fprintf('========================================================================\n\n');
    fprintf('Final Dataset Summary:\n');
    fprintf('  Total probes: %d\n', height(dataset_men));
    fprintf('  Total spectra: %d\n', sum(dataset_men.NumTotalSpectra));
    fprintf('  WHO-1: %d probes\n', sum(dataset_men.WHO_Grade == 'WHO-1'));
    fprintf('  WHO-2: %d probes\n', sum(dataset_men.WHO_Grade == 'WHO-2'));
    fprintf('  WHO-3: %d probes\n', sum(dataset_men.WHO_Grade == 'WHO-3'));
    fprintf('\nPreprocessing:\n');
    fprintf('  PP1 (Standard): %d wavenumbers per spectrum\n', size(dataset_men.CombinedSpectra_PP1{1}, 2));
    fprintf('  PP2 (Enhanced): %d wavenumbers per spectrum\n', size(dataset_men.CombinedSpectra_PP2{1}, 2));
    fprintf('\n');
end

%% Step 6: Save output
if save_output
    output_file = fullfile(data_dir, 'dataset_complete.mat');
    save(output_file, 'dataset_men', '-v7.3');
    if verbose
        fprintf('Dataset saved to: %s\n\n', output_file);
    end
end

end

%% Helper Functions

function who_str = format_who_grade(who_val)
    % Format WHO grade to standard string
    who_str = '';
    val_check = '';
    
    if isnumeric(who_val)
        val_check = num2str(who_val);
    elseif isstring(who_val) || ischar(who_val)
        val_check = char(who_val);
    elseif iscell(who_val) && ~isempty(who_val)
        val_check = char(who_val{1});
    elseif iscategorical(who_val)
        if isundefined(who_val)
            val_check = '';
        else
            val_check = char(who_val);
        end
    end
    
    % Remove trailing count notation like (94)
    val_check = regexprep(val_check, '\s*\(\d+\)$', '');
    
    % Try numeric conversion
    val_num = str2double(val_check);
    if ~isnan(val_num)
        if val_num == 1
            who_str = 'WHO-1';
        elseif val_num == 2
            who_str = 'WHO-2';
        elseif val_num == 3
            who_str = 'WHO-3';
        end
    else
        % Try string matching
        if strcmpi(strtrim(val_check), 'WHO-1')
            who_str = 'WHO-1';
        elseif strcmpi(strtrim(val_check), 'WHO-2')
            who_str = 'WHO-2';
        elseif strcmpi(strtrim(val_check), 'WHO-3')
            who_str = 'WHO-3';
        end
    end
end

function sex_str = format_sex(sex_val)
    % Format sex to standard string
    sex_str = 'Unknown';  % Default for missing values
    val_check = '';
    
    if ischar(sex_val)
        val_check = strtrim(lower(sex_val));
    elseif iscellstr(sex_val) && ~isempty(sex_val)
        val_check = strtrim(lower(sex_val{1}));
    elseif isstring(sex_val) && strlength(sex_val) > 0
        val_check = strtrim(lower(char(sex_val)));
    elseif iscategorical(sex_val)
        if isundefined(sex_val)
            val_check = '';
        else
            val_check = strtrim(lower(char(sex_val)));
        end
    end
    
    if strcmp(val_check, 'w') || strcmp(val_check, 'female')
        sex_str = 'Female';
    elseif strcmp(val_check, 'm') || strcmp(val_check, 'male')
        sex_str = 'Male';
    end
end

function subtyp_str = format_subtyp(subtyp_val)
    % Format subtype to standard abbreviation
    subtyp_str = 'Unknown';  % Default for missing values
    val_check = '';
    
    % Subtype mapping
    subtyp_map = containers.Map(...
        {'fibromatös', 'meningothelial', 'transitional', 'klarzellig', ...
         'chordoid', 'anaplastisch', 'atypisch', 'psammomatös'}, ...
        {'fibro', 'meningo', 'trans', 'clear', 'chord', 'anap', 'atyp', 'psamm'});
    
    if ischar(subtyp_val)
        val_check = strtrim(subtyp_val);
    elseif iscellstr(subtyp_val) && ~isempty(subtyp_val)
        val_check = strtrim(subtyp_val{1});
    elseif isstring(subtyp_val) && strlength(subtyp_val) > 0
        val_check = strtrim(char(subtyp_val));
    elseif iscategorical(subtyp_val)
        if isundefined(subtyp_val)
            val_check = '';
        else
            val_check = strtrim(char(subtyp_val));
        end
    end
    
    if isKey(subtyp_map, val_check)
        subtyp_str = subtyp_map(val_check);
    else
        if ~isempty(val_check) && ~any(strcmpi(val_check, {'<missing>', 'NaN', ''}))
            subtyp_str = val_check;  % Use as-is if not in map
        end
        % Otherwise remains 'Unknown' (default)
    end
end
