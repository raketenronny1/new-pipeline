
%% PHASE 0: QUALITY CONTROL AND DATA VALIDATION
% This script performs multi-level quality control on FT-IR spectroscopy data
% for meningioma classification.

function quality_control_analysis(cfg)
    clear; clc;
    addpath('src/meningioma_ftir_pipeline');

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
qc_results.train = process_sample_set(dataTableTrain, wavenumbers_roi, 'Training');

% Process test set
fprintf('Processing test set...\n');
qc_results.test = process_sample_set(dataTableTest, wavenumbers_roi, 'Test');


    %% Save QC Results
    save(fullfile(cfg.paths.qc, 'qc_flags.mat'), 'qc_results');

    % Generate QC report
    generate_qc_report(qc_results, cfg.paths.qc);
end

%% Helper Functions

function results = process_sample_set(dataTable, wavenumbers, setName)
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
    results.sample_metrics.n_Final = zeros(n_samples, 1);
    results.sample_metrics.Within_Corr = zeros(n_samples, 1);
    results.sample_metrics.Outlier_Flag = false(n_samples, 1);
    
    % Process each sample
    for i = 1:n_samples
        spectra = dataTable.CombinedSpectra{i};
        n_spectra = size(spectra, 1);
        results.sample_metrics.n_Original(i) = n_spectra;
        
        % Apply QC filters
        valid_spectra = true(n_spectra, 1);
        
        % 1.1 SNR Check
        signal_region = find(wavenumbers >= 1000 & wavenumbers <= 1700);
        noise_region = find(wavenumbers >= 1750 & wavenumbers <= 1800);
        
        snr = zeros(n_spectra, 1);
        for j = 1:n_spectra
            signal = max(spectra(j,signal_region)) - min(spectra(j,signal_region));
            noise = std(spectra(j,noise_region));
            snr(j) = signal / noise;
        end
        valid_spectra = valid_spectra & (snr >= 10);
        results.sample_metrics.n_After_SNR(i) = sum(valid_spectra);
        
        % 1.2 Saturation Check
        max_abs = max(spectra, [], 2);
        valid_spectra = valid_spectra & (max_abs <= 1.8);
        results.sample_metrics.n_After_Saturation(i) = sum(valid_spectra);
        
        % 1.3 Baseline Quality
        baseline_region_1 = find(wavenumbers >= 950 & wavenumbers <= 1000);
        baseline_region_2 = find(wavenumbers >= 1750 & wavenumbers <= 1800);
        
        baseline_sd = zeros(n_spectra, 1);
        for j = 1:n_spectra
            baseline_values = [spectra(j,baseline_region_1), spectra(j,baseline_region_2)];
            baseline_sd(j) = std(baseline_values);
        end
        valid_spectra = valid_spectra & (baseline_sd <= 0.02);
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
        valid_spectra = valid_spectra & (peak_ratio >= 1.2 & peak_ratio <= 3.5);
        results.sample_metrics.n_After_Amide(i) = sum(valid_spectra);
        
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

function generate_qc_report(qc_results, output_dir)
    % Create QC summary table
    qc_metrics_train = qc_results.train.sample_metrics;
    qc_metrics_test = qc_results.test.sample_metrics;
    
    % Write QC metrics to CSV
    writetable(qc_metrics_train, fullfile(output_dir, 'qc_metrics_train.csv'));
    writetable(qc_metrics_test, fullfile(output_dir, 'qc_metrics_test.csv'));
    
    % Generate visualizations
    
    % 1. SNR Distribution
    figure('Position', [100, 100, 800, 600]);
    histogram(qc_metrics_train.n_After_SNR ./ qc_metrics_train.n_Original * 100);
    xlabel('Spectra Retained After SNR Check (%)');
    ylabel('Number of Samples');
    title('SNR Filter Impact (Training Set)');
    saveas(gcf, fullfile(output_dir, 'qc_snr_distribution.png'));
    
    % 2. Within-sample Correlation
    figure('Position', [100, 100, 800, 600]);
    boxplot([qc_metrics_train.Within_Corr; qc_metrics_test.Within_Corr], ...
            [repmat({'Training'}, height(qc_metrics_train), 1); ...
             repmat({'Test'}, height(qc_metrics_test), 1)]);
    ylabel('Within-sample Correlation');
    title('Spectral Consistency');
    saveas(gcf, fullfile(output_dir, 'qc_correlation_boxplot.png'));
    
    % 3. Spectra Retention
    figure('Position', [100, 100, 1000, 600]);
    bar([qc_metrics_train.n_Original, qc_metrics_train.n_Final]);
    xlabel('Sample Index');
    ylabel('Number of Spectra');
    title('Training Set: Original vs Retained Spectra');
    legend({'Original', 'After QC'});
    saveas(gcf, fullfile(output_dir, 'qc_spectra_retention.png'));
    
    close all;
end