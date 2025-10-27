
%% PHASE 0: QUALITY CONTROL AND DATA VALIDATION
% This script performs multi-level quality control on FT-IR spectroscopy data
% for meningioma classification.

function quality_control_analysis(cfg)
    % Input validation
    if ~isstruct(cfg) || ~isfield(cfg, 'paths') || ~isfield(cfg.paths, 'qc')
        error('Invalid cfg structure. Must contain paths.qc');
    end
    
    % Validate QC parameters
    if ~isfield(cfg, 'qc')
        error('Configuration must include qc parameters');
    end
    required_qc_fields = {'snr_threshold', 'max_absorbance', 'baseline_sd_threshold', ...
                         'amide_ratio_min', 'amide_ratio_max'};
    for i = 1:length(required_qc_fields)
        if ~isfield(cfg.qc, required_qc_fields{i})
            error('Configuration missing required QC parameter: %s', required_qc_fields{i});
        end
    end

    % Create results directory if it doesn't exist
    if ~exist(cfg.paths.qc, 'dir')
        mkdir(cfg.paths.qc);
    end

    %% Load Data
    fprintf('Loading data...\n');
    load(fullfile(cfg.paths.data, 'data_table_train.mat'), 'dataTableTrain');
    load(fullfile(cfg.paths.data, 'data_table_test.mat'), 'dataTableTest');
    load(fullfile(cfg.paths.data, 'wavenumbers.mat'), 'wavenumbers_roi');

    %% LEVEL 1: SPECTRUM-LEVEL QUALITY CONTROL

    % Initialize QC results structure
    qc_results = struct();
    qc_results.train = struct();
    qc_results.test = struct();

    % Process training set
    fprintf('Processing training set...\n');
    qc_results.train = process_sample_set(dataTableTrain, wavenumbers_roi, 'Training', cfg);

% Process test set
fprintf('Processing test set...\n');
qc_results.test = process_sample_set(dataTableTest, wavenumbers_roi, 'Test', cfg);


    %% Save QC Results
    save(fullfile(cfg.paths.qc, 'qc_flags.mat'), 'qc_results');
    
    % Save rejected spectra workspace for easy plotting
    save_rejected_spectra_workspace(qc_results, dataTableTrain, dataTableTest, wavenumbers_roi, cfg.paths.qc);

    % Generate QC report and plots
    generate_qc_report(qc_results, cfg.paths.qc);
    
    % Plot rejected spectra
    plot_rejected_spectra(qc_results, dataTableTrain, dataTableTest, wavenumbers_roi, cfg.paths.qc);
end

%% Helper Functions

function results = process_sample_set(dataTable, wavenumbers, setName, cfg)
    results = struct();
    n_samples = height(dataTable);
    
    % Initialize arrays for storing QC metrics
    results.sample_metrics = table();
    results.sample_metrics.Diss_ID = dataTable.Diss_ID;
    results.sample_metrics.WHO_Grade = dataTable.WHO_Grade;
    results.sample_metrics.n_Original = zeros(n_samples, 1);
    results.sample_metrics.n_After_SNR = zeros(n_samples, 1);
    results.sample_metrics.n_After_Saturation = zeros(n_samples, 1);
    results.sample_metrics.n_After_Baseline = zeros(n_samples, 1);
    results.sample_metrics.n_After_Amide = zeros(n_samples, 1);
    results.sample_metrics.n_After_Mahalanobis = zeros(n_samples, 1);
    results.sample_metrics.n_Final = zeros(n_samples, 1);
    results.sample_metrics.Within_Corr = zeros(n_samples, 1);
    results.sample_metrics.Outlier_Flag = false(n_samples, 1);
    
    % Initialize detailed rejection tracking
    results.rejected_spectra = table();
    rejected_list = {};
    
    % Process each sample
    for i = 1:n_samples
        % Get spectra for this sample and ensure correct orientation
        spectra = dataTable.CombinedSpectra{i};
        [n_rows, n_cols] = size(spectra);
        
        % Ensure spectra are in rows and wavelengths in columns
        if n_cols ~= length(wavenumbers)
            spectra = spectra';
            n_rows = size(spectra, 1);
        end
        
        n_spectra = n_rows;
        results.sample_metrics.n_Original(i) = n_spectra;
        
        % Apply QC filters
        valid_spectra = true(n_spectra, 1);
        
        % 1.1 SNR Check
        signal_region = find(wavenumbers >= 1000 & wavenumbers <= 1700);
        noise_region = find(wavenumbers >= 1750 & wavenumbers <= 1800);
        
        % Validate that the regions exist in our wavenumber range
        if isempty(signal_region) || isempty(noise_region)
            warning('Wavenumber regions for SNR calculation not found in data range');
            signal_region = 1:floor(length(wavenumbers)/2);  % Use first half for signal
            noise_region = (floor(length(wavenumbers)/2)+1):length(wavenumbers);  % Use second half for noise
        end
        
        snr = zeros(n_spectra, 1);
        for j = 1:n_spectra
            signal = max(spectra(j,signal_region)) - min(spectra(j,signal_region));
            noise = std(spectra(j,noise_region));
            % Avoid division by zero
            if noise == 0
                snr(j) = 0;  % Mark as poor SNR if no noise variation
            else
                snr(j) = signal / noise;
            end
            % Debug output for SNR calculation
            if i == 1 && j == 1
                fprintf('Debug - Sample 1, Spectrum 1:\n');
                fprintf('Signal: %.4f, Noise: %.4f, SNR: %.4f\n', signal, noise, snr(j));
            end
        end
        
        % Track rejected spectra from SNR filter
        snr_failed = valid_spectra & (snr < cfg.qc.snr_threshold);
        rejected_list = track_rejections(rejected_list, snr_failed, i, dataTable.Diss_ID{i}, ...
                                        dataTable.Patient_ID{i}, 'SNR', snr);
        
        valid_spectra = valid_spectra & (snr >= cfg.qc.snr_threshold);
        results.sample_metrics.n_After_SNR(i) = sum(valid_spectra);
        
        % 1.2 Saturation Check
        max_abs = max(spectra, [], 2);
        if i == 1
            fprintf('Debug - Sample 1: Max absorbance: %.4f\n', max_abs(1));
        end
        
        % Track rejected spectra from saturation filter
        saturation_failed = valid_spectra & (max_abs > cfg.qc.max_absorbance);
        rejected_list = track_rejections(rejected_list, saturation_failed, i, dataTable.Diss_ID{i}, ...
                                        dataTable.Patient_ID{i}, 'Saturation', max_abs);
        
        valid_spectra = valid_spectra & (max_abs <= cfg.qc.max_absorbance);
        results.sample_metrics.n_After_Saturation(i) = sum(valid_spectra);
        
        % 1.3 Baseline Quality
        baseline_region_1 = find(wavenumbers >= 950 & wavenumbers <= 1000);
        baseline_region_2 = find(wavenumbers >= 1750 & wavenumbers <= 1800);
        
        % Validate that the baseline regions exist in our wavenumber range
        if isempty(baseline_region_1) || isempty(baseline_region_2)
            warning('Baseline regions not found in data range');
            % Use first and last 10% of spectrum for baseline
            n_points = floor(length(wavenumbers) * 0.1);
            baseline_region_1 = 1:n_points;
            baseline_region_2 = (length(wavenumbers)-n_points+1):length(wavenumbers);
        end
        
        baseline_sd = zeros(n_spectra, 1);
        for j = 1:n_spectra
            baseline_values = [spectra(j,baseline_region_1), spectra(j,baseline_region_2)];
            baseline_sd(j) = std(baseline_values);
        end
        if i == 1
            fprintf('Debug - Sample 1: Baseline SD: %.4f\n', baseline_sd(1));
        end
        
        % Track rejected spectra from baseline filter
        baseline_failed = valid_spectra & (baseline_sd > cfg.qc.baseline_sd_threshold);
        rejected_list = track_rejections(rejected_list, baseline_failed, i, dataTable.Diss_ID{i}, ...
                                        dataTable.Patient_ID{i}, 'Baseline', baseline_sd);
        
        valid_spectra = valid_spectra & (baseline_sd <= cfg.qc.baseline_sd_threshold);
        results.sample_metrics.n_After_Baseline(i) = sum(valid_spectra);
        
        % 1.4 & 1.5 Amide Peaks and Ratio Check
        amide_I_region = find(wavenumbers >= 1630 & wavenumbers <= 1670);
        amide_II_region = find(wavenumbers >= 1530 & wavenumbers <= 1570);
        
        peak_ratio = zeros(n_spectra, 1);
        for j = 1:n_spectra
            amide_I = max(spectra(j,amide_I_region));
            amide_II = max(spectra(j,amide_II_region));
            peak_ratio(j) = amide_I / amide_II;
        end
        if i == 1
            fprintf('Debug - Sample 1: Peak ratio: %.4f\n', peak_ratio(1));
        end
        
        % Track rejected spectra from amide ratio filter
        amide_failed = valid_spectra & ~(peak_ratio >= cfg.qc.amide_ratio_min & peak_ratio <= cfg.qc.amide_ratio_max);
        rejected_list = track_rejections(rejected_list, amide_failed, i, dataTable.Diss_ID{i}, ...
                                        dataTable.Patient_ID{i}, 'AmideRatio', peak_ratio);
        
        valid_spectra = valid_spectra & (peak_ratio >= cfg.qc.amide_ratio_min & peak_ratio <= cfg.qc.amide_ratio_max);
        results.sample_metrics.n_After_Amide(i) = sum(valid_spectra);
        
        % 1.6 Mahalanobis Distance Outlier Detection (spectrum-level)
        % Only apply if we have enough spectra after basic QC
        if sum(valid_spectra) > 20  % Need sufficient spectra for PCA
            valid_specs_temp = spectra(valid_spectra, :);
            
            % Compute PCA on valid spectra for this sample
            try
                [~, score] = pca(valid_specs_temp);
                
                % Use first few PCs for outlier detection
                n_pcs = min(10, size(score, 2));
                scores_subset = score(:, 1:n_pcs);
                
                % Compute Mahalanobis distance
                mahal_dist = mahal(scores_subset, scores_subset);
                
                % Chi-squared threshold from config
                threshold = chi2inv(cfg.qc.outlier_confidence, n_pcs);
                
                % Identify outliers in the valid subset
                outlier_in_valid = mahal_dist > threshold;
                
                % Map back to full spectrum indices and track rejections
                valid_indices = find(valid_spectra);
                mahal_failed_indices = false(n_spectra, 1);
                mahal_failed_indices(valid_indices(outlier_in_valid)) = true;
                
                mahal_distances = zeros(n_spectra, 1);
                mahal_distances(valid_indices) = mahal_dist;
                rejected_list = track_rejections(rejected_list, mahal_failed_indices, i, dataTable.Diss_ID{i}, ...
                                                dataTable.Patient_ID{i}, 'Mahalanobis', mahal_distances);
                
                % Update valid spectra
                valid_spectra(valid_indices(outlier_in_valid)) = false;
            catch ME
                % If PCA fails, skip Mahalanobis for this sample
                warning('Sample %d: Mahalanobis filtering skipped - %s', i, ME.message);
            end
        end
        
        % Store how many were filtered by Mahalanobis
        results.sample_metrics.n_After_Mahalanobis(i) = sum(valid_spectra);
        
        % Calculate final valid spectra count
        results.sample_metrics.n_Final(i) = sum(valid_spectra);
        
        % Calculate within-sample correlation for valid spectra
        valid_specs = spectra(valid_spectra,:);
        if size(valid_specs, 1) > 1
            corr_matrix = corrcoef(valid_specs');
            results.sample_metrics.Within_Corr(i) = mean(corr_matrix(triu(true(size(corr_matrix)),1)));
        end
        
        % Store valid spectra mask for this sample
        results.valid_spectra_masks{i} = valid_spectra;
    end
    
    % Convert rejection tracking list to table
    if ~isempty(rejected_list)
        % Convert cell array of structs to struct array, then to table
        rejected_struct_array = vertcat(rejected_list{:});
        results.rejected_spectra = struct2table(rejected_struct_array);
    else
        % Create empty table with correct structure
        results.rejected_spectra = table('Size', [0 7], ...
            'VariableTypes', {'double', 'cell', 'cell', 'double', 'cell', 'double', 'double'}, ...
            'VariableNames', {'Sample_Index', 'Diss_ID', 'Patient_ID', 'Spectrum_Index', ...
                              'Rejection_Reason', 'QC_Value', 'Sample_WHO_Grade'});
    end
    
    % Level 3: Cross-sample outlier detection
    % Compute PCA on valid representative spectra
    X_representatives = zeros(n_samples, size(wavenumbers, 2));
    for i = 1:n_samples
        valid_specs = dataTable.CombinedSpectra{i}(results.valid_spectra_masks{i},:);
        if ~isempty(valid_specs)
            X_representatives(i,:) = mean(valid_specs, 1);
        end
    end
    
    % Separate WHO grades for outlier detection
    who1_idx = dataTable.WHO_Grade == 'WHO-1';
    who3_idx = dataTable.WHO_Grade == 'WHO-3';
    
    % Detect outliers within each WHO grade
    results.sample_metrics.Outlier_Flag(who1_idx) = detect_outliers(X_representatives(who1_idx,:));
    results.sample_metrics.Outlier_Flag(who3_idx) = detect_outliers(X_representatives(who3_idx,:));
    
    fprintf('%s set processing complete.\n', setName);
end

function outliers = detect_outliers(X)
    % Compute PCA
    [~, score] = pca(X);
    
    % Use first few PCs for outlier detection
    n_pcs = min(10, size(score, 2));
    scores_subset = score(:, 1:n_pcs);
    
    % Compute Mahalanobis distance
    mahal_dist = mahal(scores_subset, scores_subset);
    
    % Chi-squared threshold (99% confidence)
    threshold = chi2inv(0.99, n_pcs);
    
    % Identify outliers
    outliers = mahal_dist > threshold;
end

function rejected_list = track_rejections(rejected_list, failed_mask, sample_idx, diss_id, patient_id, reason, qc_values)
    % Track which spectra were rejected and why
    % Inputs:
    %   rejected_list - cell array of existing rejection records
    %   failed_mask - logical array indicating which spectra failed this criterion
    %   sample_idx - index of the sample being processed
    %   diss_id - sample identifier
    %   patient_id - patient identifier  
    %   reason - string describing the QC criterion (e.g., 'SNR', 'Baseline')
    %   qc_values - actual QC metric values for all spectra
    
    % Get indices of failed spectra
    failed_indices = find(failed_mask);
    
    % Record each failed spectrum
    for j = 1:length(failed_indices)
        spec_idx = failed_indices(j);
        record = struct();
        record.Sample_Index = sample_idx;
        record.Diss_ID = {diss_id};
        record.Patient_ID = {patient_id};
        record.Spectrum_Index = spec_idx;
        record.Rejection_Reason = {reason};
        record.QC_Value = qc_values(spec_idx);
        record.Sample_WHO_Grade = NaN;  % Will be filled from sample_metrics later if needed
        
        rejected_list{end+1} = record;
    end
end

function generate_qc_report(qc_results, output_dir)
    % Create QC summary table
    qc_metrics_train = qc_results.train.sample_metrics;
    qc_metrics_test = qc_results.test.sample_metrics;
    
    % Write QC metrics to CSV
    writetable(qc_metrics_train, fullfile(output_dir, 'qc_metrics_train.csv'));
    writetable(qc_metrics_test, fullfile(output_dir, 'qc_metrics_test.csv'));
    
    % Write rejected spectra logs
    if ~isempty(qc_results.train.rejected_spectra)
        writetable(qc_results.train.rejected_spectra, fullfile(output_dir, 'qc_rejected_spectra_train.csv'));
        fprintf('Saved rejection log for training set: %d rejected spectra\n', ...
                height(qc_results.train.rejected_spectra));
    end
    if ~isempty(qc_results.test.rejected_spectra)
        writetable(qc_results.test.rejected_spectra, fullfile(output_dir, 'qc_rejected_spectra_test.csv'));
        fprintf('Saved rejection log for test set: %d rejected spectra\n', ...
                height(qc_results.test.rejected_spectra));
    end
    
    % Print summary statistics
    fprintf('\n=== QC Summary Statistics ===\n');
    fprintf('Training Set:\n');
    fprintf('  Total spectra: %d\n', sum(qc_metrics_train.n_Original));
    fprintf('  After SNR: %d (%.1f%%)\n', sum(qc_metrics_train.n_After_SNR), ...
        100*sum(qc_metrics_train.n_After_SNR)/sum(qc_metrics_train.n_Original));
    fprintf('  After Saturation: %d (%.1f%%)\n', sum(qc_metrics_train.n_After_Saturation), ...
        100*sum(qc_metrics_train.n_After_Saturation)/sum(qc_metrics_train.n_Original));
    fprintf('  After Baseline: %d (%.1f%%)\n', sum(qc_metrics_train.n_After_Baseline), ...
        100*sum(qc_metrics_train.n_After_Baseline)/sum(qc_metrics_train.n_Original));
    fprintf('  After Amide: %d (%.1f%%)\n', sum(qc_metrics_train.n_After_Amide), ...
        100*sum(qc_metrics_train.n_After_Amide)/sum(qc_metrics_train.n_Original));
    if isfield(qc_metrics_train, 'n_After_Mahalanobis')
        fprintf('  After Mahalanobis: %d (%.1f%%)\n', sum(qc_metrics_train.n_After_Mahalanobis), ...
            100*sum(qc_metrics_train.n_After_Mahalanobis)/sum(qc_metrics_train.n_Original));
    end
    fprintf('  Final valid: %d (%.1f%%)\n', sum(qc_metrics_train.n_Final), ...
        100*sum(qc_metrics_train.n_Final)/sum(qc_metrics_train.n_Original));
    fprintf('  Samples flagged as outliers: %d\n\n', sum(qc_metrics_train.Outlier_Flag));
    
    % Generate visualizations
    
    % 1. SNR Distribution
    fig1 = figure('Position', [100, 100, 800, 600], 'Visible', 'off');
    histogram(qc_metrics_train.n_After_SNR ./ qc_metrics_train.n_Original * 100);
    xlabel('Spectra Retained After SNR Check (%)');
    ylabel('Number of Samples');
    title('SNR Filter Impact (Training Set)');
    saveas(fig1, fullfile(output_dir, 'qc_snr_distribution.png'));
    close(fig1);
    
    % 2. Within-sample Correlation
    fig2 = figure('Position', [100, 100, 800, 600], 'Visible', 'off');
    boxplot([qc_metrics_train.Within_Corr; qc_metrics_test.Within_Corr], ...
            [repmat({'Training'}, height(qc_metrics_train), 1); ...
             repmat({'Test'}, height(qc_metrics_test), 1)]);
    ylabel('Within-sample Correlation');
    title('Spectral Consistency');
    saveas(fig2, fullfile(output_dir, 'qc_correlation_boxplot.png'));
    close(fig2);
    
    % 3. Spectra Retention
    fig3 = figure('Position', [100, 100, 1000, 600], 'Visible', 'off');
    bar([qc_metrics_train.n_Original, qc_metrics_train.n_Final]);
    xlabel('Sample Index');
    ylabel('Number of Spectra');
    title('Training Set: Original vs Retained Spectra');
    legend({'Original', 'After QC'});
    saveas(fig3, fullfile(output_dir, 'qc_spectra_retention.png'));
    close(fig3);
end


function save_rejected_spectra_workspace(qc_results, dataTableTrain, dataTableTest, wavenumbers, output_dir)
    % Save rejected spectra as a convenient workspace variable for manual plotting
    % Creates a structure that's easy to explore and visualize
    
    fprintf('Saving rejected spectra workspace...\n');
    
    rejected_spectra_workspace = struct();
    rejected_spectra_workspace.wavenumbers = wavenumbers;
    rejected_spectra_workspace.description = 'Rejected spectra organized by dataset, sample, and reason';
    
    % Process train and test sets
    datasets = {'train', 'test'};
    dataTables = {dataTableTrain, dataTableTest};
    
    for d = 1:length(datasets)
        dataset_name = datasets{d};
        dataTable = dataTables{d};
        rejected_table = qc_results.(dataset_name).rejected_spectra;
        
        rejected_spectra_workspace.(dataset_name) = struct();
        rejected_spectra_workspace.(dataset_name).rejection_table = rejected_table;
        rejected_spectra_workspace.(dataset_name).spectra = {};
        
        if isempty(rejected_table) || height(rejected_table) == 0
            continue;
        end
        
        % Extract actual spectral data for each rejected spectrum
        for i = 1:height(rejected_table)
            sample_idx = rejected_table.Sample_Index(i);
            spectrum_idx = rejected_table.Spectrum_Index(i);
            
            % Get spectrum
            all_spectra = dataTable.CombinedSpectra{sample_idx};
            if size(all_spectra, 2) ~= length(wavenumbers)
                all_spectra = all_spectra';
            end
            
            % Store spectrum with metadata
            entry = struct();
            entry.Diss_ID = rejected_table.Diss_ID{i};
            entry.Patient_ID = rejected_table.Patient_ID{i};
            entry.Spectrum_Index = spectrum_idx;
            entry.Rejection_Reason = rejected_table.Rejection_Reason{i};
            entry.QC_Value = rejected_table.QC_Value(i);
            entry.spectrum = all_spectra(spectrum_idx, :);
            
            rejected_spectra_workspace.(dataset_name).spectra{end+1} = entry;
        end
    end
    
    % Save to file
    save(fullfile(output_dir, 'rejected_spectra_workspace.mat'), 'rejected_spectra_workspace');
    
    fprintf('✓ Saved: rejected_spectra_workspace.mat\n');
    fprintf('  Usage example:\n');
    fprintf('    load(''rejected_spectra_workspace.mat'')\n');
    fprintf('    plot(rejected_spectra_workspace.wavenumbers, ...\n');
    fprintf('         rejected_spectra_workspace.train.spectra{1}.spectrum)\n\n');
end


function plot_rejected_spectra(qc_results, dataTableTrain, dataTableTest, wavenumbers, output_dir)
    % Plot rejected spectra organized by patient-sample and rejection reason
    % Creates tiled layouts showing rejected spectra vs wavenumbers
    
    fprintf('\n=== Plotting Rejected Spectra ===\n');
    
    % Process training and test sets
    datasets = {'train', 'test'};
    dataTables = {dataTableTrain, dataTableTest};
    
    for d = 1:length(datasets)
        dataset_name = datasets{d};
        dataTable = dataTables{d};
        rejected_table = qc_results.(dataset_name).rejected_spectra;
        
        if isempty(rejected_table) || height(rejected_table) == 0
            fprintf('  No rejected spectra in %s set\n', dataset_name);
            continue;
        end
        
        fprintf('  Plotting %d rejected spectra from %s set...\n', ...
                height(rejected_table), dataset_name);
        
        % Get unique rejection reasons
        reasons = unique(rejected_table.Rejection_Reason);
        
        % Create one figure per rejection reason
        for r = 1:length(reasons)
            reason = reasons{r};
            reason_mask = strcmp(rejected_table.Rejection_Reason, reason);
            reason_data = rejected_table(reason_mask, :);
            
            % Limit to first 20 samples to keep figure manageable
            n_plot = min(20, height(reason_data));
            
            if n_plot == 0
                continue;
            end
            
            % Create figure with tiled layout
            fig = figure('Position', [100, 100, 1400, 900], 'Visible', 'off');
            t = tiledlayout(4, 5, 'Padding', 'compact', 'TileSpacing', 'compact');
            title(t, sprintf('%s Set - Rejected Spectra: %s (n=%d, showing first %d)', ...
                   upper(dataset_name), reason, height(reason_data), n_plot), ...
                   'FontSize', 14, 'FontWeight', 'bold');
            
            % Plot each rejected spectrum
            for i = 1:n_plot
                sample_idx = reason_data.Sample_Index(i);
                spectrum_idx = reason_data.Spectrum_Index(i);
                diss_id = reason_data.Diss_ID{i};
                patient_id = reason_data.Patient_ID{i};
                qc_value = reason_data.QC_Value(i);
                
                % Get the spectrum
                all_spectra = dataTable.CombinedSpectra{sample_idx};
                
                % Ensure correct orientation
                if size(all_spectra, 2) ~= length(wavenumbers)
                    all_spectra = all_spectra';
                end
                
                spectrum = all_spectra(spectrum_idx, :);
                
                % Plot
                nexttile;
                plot(wavenumbers, spectrum, 'LineWidth', 1.2);
                xlabel('Wavenumber (cm^{-1})', 'FontSize', 8);
                ylabel('Absorbance', 'FontSize', 8);
                title(sprintf('%s (P:%s)\n%s=%.2f', diss_id, patient_id, reason, qc_value), ...
                      'FontSize', 9, 'Interpreter', 'none');
                grid on;
                set(gca, 'FontSize', 8);
            end
            
            % Save figure
            filename = fullfile(output_dir, sprintf('rejected_spectra_%s_%s.png', ...
                                dataset_name, reason));
            saveas(fig, filename);
            close(fig);
            
            fprintf('    Saved: %s\n', filename);
        end
        
        % Create summary figure showing distribution of rejected spectra by sample
        fig_summary = figure('Position', [100, 100, 1200, 600], 'Visible', 'off');
        
        % Count rejections per sample
        [unique_samples, ~, sample_groups] = unique(rejected_table.Sample_Index);
        rejection_counts = accumarray(sample_groups, 1);
        
        % Get Diss_IDs for x-axis labels
        diss_ids = cell(length(unique_samples), 1);
        for i = 1:length(unique_samples)
            diss_ids{i} = rejected_table.Diss_ID{find(rejected_table.Sample_Index == unique_samples(i), 1)};
        end
        
        % Plot
        bar(rejection_counts);
        xlabel('Sample (Diss\_ID)', 'Interpreter', 'tex');
        ylabel('Number of Rejected Spectra');
        title(sprintf('%s Set: Rejected Spectra Count per Sample', upper(dataset_name)));
        xticks(1:length(unique_samples));
        xticklabels(diss_ids);
        xtickangle(45);
        grid on;
        
        % Save
        saveas(fig_summary, fullfile(output_dir, sprintf('rejected_summary_%s.png', dataset_name)));
        close(fig_summary);
    end
    
    fprintf('✓ Rejected spectra plots saved\n');
end