function eda_results = exploratory_data_analysis_no_plots(dataset_men, train_indices)
% EDA WITHOUT PLOTS - for batch mode execution
% Computes PCA and outlier detection only, skips all visualizations

pp_type = 'PP1';
output_dir = 'results/eda';
t2_threshold = 3;
q_threshold = 3;

% Create output directory
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf('PCA-based outlier detection (no plots)...\n');

%% Load wavenumbers
load('data/wavenumbers.mat', 'wavenumbers_roi');
wavenumbers = wavenumbers_roi;

%% Build spectrum-level data matrix
spectra_col = 'CombinedSpectra_PP1';
n_probes = height(dataset_men);
all_spectra = [];
all_who_grades = [];
all_probe_ids = [];  % This will store row indices
all_probe_uids = [];  % This will store actual ProbeUIDs
all_is_train = [];

fprintf('  Building data matrix from %d probes...\n', n_probes);

for i = 1:n_probes
    spectra_matrix = dataset_men.(spectra_col){i};
    n_spectra_probe = size(spectra_matrix, 1);
    
    all_spectra = [all_spectra; spectra_matrix];
    all_who_grades = [all_who_grades; repmat(dataset_men.WHO_Grade(i), n_spectra_probe, 1)];
    all_probe_ids = [all_probe_ids; repmat(i, n_spectra_probe, 1)];
    
    % Store actual ProbeUID if available
    if ismember('ProbeUID', dataset_men.Properties.VariableNames)
        all_probe_uids = [all_probe_uids; repmat(dataset_men.ProbeUID(i), n_spectra_probe, 1)];
    else
        all_probe_uids = [all_probe_uids; repmat(i, n_spectra_probe, 1)];  % Fallback to index
    end
    
    is_train_probe = train_indices(i);
    all_is_train = [all_is_train; repmat(is_train_probe, n_spectra_probe, 1)];
    
    if mod(i, 10) == 0
        fprintf('    Processed %d/%d probes\r', i, n_probes);
    end
end

fprintf('\n  Total spectra: %d\n', size(all_spectra, 1));
fprintf('  WHO-1: %d, WHO-3: %d\n', ...
        sum(all_who_grades == 'WHO-1'), sum(all_who_grades == 'WHO-3'));

%% PCA for outlier detection (training set only)
fprintf('  Computing PCA on training set...\n');

pca_mask = all_is_train & ((all_who_grades == 'WHO-1') | (all_who_grades == 'WHO-3'));
X_pca = all_spectra(pca_mask, :);
who_grades_pca = all_who_grades(pca_mask);
probe_ids_pca = all_probe_ids(pca_mask);  % Row indices
probe_uids_pca = all_probe_uids(pca_mask);  % Actual ProbeUIDs

fprintf('    PCA spectra: %d (WHO-1: %d, WHO-3: %d)\n', ...
        size(X_pca, 1), sum(who_grades_pca == 'WHO-1'), sum(who_grades_pca == 'WHO-3'));

% Mean-center
X_mean = mean(X_pca, 1);
X_centered = X_pca - X_mean;

% PCA
fprintf('    Running PCA...\n');
[coeff, score, latent, ~, explained] = pca(X_centered);

fprintf('    PC1: %.2f%%, PC2: %.2f%%, PC1-3: %.2f%%\n', ...
        explained(1), explained(2), sum(explained(1:3)));

%% Calculate outlier statistics
n_pcs = min(15, size(score, 2));  % Use 15 PCs for outlier detection
fprintf('    Computing outlier statistics (%d PCs)...\n', n_pcs);

% T² statistic
T2 = zeros(size(score, 1), 1);
for i = 1:size(score, 1)
    T2(i) = sum((score(i, 1:n_pcs).^2) ./ latent(1:n_pcs)');
end

% Q statistic (SPE)
X_reconstructed = score(:, 1:n_pcs) * coeff(:, 1:n_pcs)';
residuals = X_centered - X_reconstructed;
Q = sum(residuals.^2, 2);

% Thresholds
T2_limit = t2_threshold * mean(T2) + t2_threshold * std(T2);
Q_limit = q_threshold * mean(Q) + q_threshold * std(Q);

% Identify outliers
outliers_T2 = T2 > T2_limit;
outliers_Q = Q > Q_limit;
outliers_both = outliers_T2 & outliers_Q;

fprintf('    T² outliers: %d (%.1f%%)\n', sum(outliers_T2), 100*sum(outliers_T2)/length(T2));
fprintf('    Q outliers: %d (%.1f%%)\n', sum(outliers_Q), 100*sum(outliers_Q)/length(Q));
fprintf('    Both: %d (%.1f%%)\n', sum(outliers_both), 100*sum(outliers_both)/length(T2));

%% Package results
eda_results = struct();
eda_results.preprocessing_type = pp_type;
eda_results.wavenumbers = wavenumbers;

% PCA results
eda_results.pca.coeff = coeff;
eda_results.pca.score = score;
eda_results.pca.latent = latent;
eda_results.pca.explained = explained;
eda_results.pca.T2 = T2;
eda_results.pca.Q = Q;
eda_results.pca.T2_limit = T2_limit;
eda_results.pca.Q_limit = Q_limit;
eda_results.pca.outliers_T2 = outliers_T2;
eda_results.pca.outliers_Q = outliers_Q;
eda_results.pca.outliers_both = outliers_both;

% Additional info for pipeline
eda_results.probe_ids_pca = probe_ids_pca;  % Row indices (for reference)
eda_results.probe_uids_pca = probe_uids_pca;  % Actual ProbeUIDs (for matching)
eda_results.is_train = all_is_train;
eda_results.X_mean = X_mean;
eda_results.n_pcs_used = n_pcs;

% Save
fprintf('  Saving results...\n');
save(fullfile(output_dir, sprintf('eda_results_%s.mat', pp_type)), 'eda_results');

fprintf('  EDA complete! Outlier flags ready for pipeline.\n');

end
