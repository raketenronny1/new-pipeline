function eda_results = exploratory_data_analysis(dataset_men, varargin)
% EXPLORATORY_DATA_ANALYSIS Comprehensive EDA for FTIR meningioma dataset
%
% SYNTAX:
%   eda_results = exploratory_data_analysis(dataset_men)
%   eda_results = exploratory_data_analysis(dataset_men, 'Name', Value, ...)
%
% INPUTS:
%   dataset_men - Complete dataset table from prepare_ftir_dataset
%
% OPTIONAL PARAMETERS:
%   'PreprocessingType' - 'PP1' or 'PP2' (default: 'PP1')
%   'OutputDir'         - Directory for plots (default: 'results/eda')
%   'Verbose'           - Display progress (default: true)
%   'T2_Threshold'      - T² threshold multiplier (default: 3)
%   'Q_Threshold'       - Q threshold multiplier (default: 3)
%   'TrainIndices'      - Logical or numeric indices for training set samples (default: [])
%                         If empty, uses all WHO-1 and WHO-3 samples for PCA
%                         If provided, PCA only uses training set WHO-1 and WHO-3
%
% OUTPUTS:
%   eda_results - Structure with EDA results and statistics
%
% DESCRIPTION:
%   Performs comprehensive exploratory data analysis including:
%   1. Mean spectra per WHO grade with SD bands
%   2. Individual probe spectra in tiled layout
%   3. Descriptive statistics per class and wavenumber
%   4. PCA with outlier detection (T² and Q statistics)
%   5. Score plots with outlier highlighting
%   6. Loadings analysis
%   7. Outlier spectra visualization
%
% GERMAN LABELS:
%   xlabel: 'Wellenzahl (cm^{-1})'
%   ylabel: 'Absorption (a.u.)'
%
% COLORS:
%   WHO-1: [230, 153, 102]/255 (Orange)
%   WHO-2: [161, 215, 106]/255 (Green)
%   WHO-3: [102, 179, 230]/255 (Blue)
%
% See also: PREPARE_FTIR_DATASET, PCA

% Author: GitHub Copilot
% Date: 2025-10-24

%% Parse inputs
p = inputParser;
addRequired(p, 'dataset_men', @istable);
addParameter(p, 'PreprocessingType', 'PP1', @(x) ismember(x, {'PP1', 'PP2'}));
addParameter(p, 'OutputDir', 'results/eda', @ischar);
addParameter(p, 'Verbose', true, @islogical);
addParameter(p, 'T2_Threshold', 3, @isnumeric);
addParameter(p, 'Q_Threshold', 3, @isnumeric);
addParameter(p, 'TrainIndices', [], @(x) isempty(x) || islogical(x) || isnumeric(x));
addParameter(p, 'Headless', false, @islogical);  % For batch mode - creates invisible figures
parse(p, dataset_men, varargin{:});

pp_type = p.Results.PreprocessingType;
output_dir = p.Results.OutputDir;
verbose = p.Results.Verbose;
t2_threshold = p.Results.T2_Threshold;
q_threshold = p.Results.Q_Threshold;
train_indices = p.Results.TrainIndices;
headless = p.Results.Headless;

% REQUIRE train_indices - PCA should NEVER use full dataset
if isempty(train_indices)
    error('TrainIndices parameter is required! PCA must only use training set (WHO-1 and WHO-3).\nProvide train_indices as logical or numeric array indicating training set samples.');
end

% Convert train_indices to logical if numeric
if isnumeric(train_indices)
    temp = false(height(dataset_men), 1);
    temp(train_indices) = true;
    train_indices = temp;
end

%% Setup
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Define WHO colors
colorWHO1 = [230, 153, 102] / 255;  % Orange (0.9, 0.6, 0.4)
colorWHO2 = [161, 215, 106] / 255;  % Green (0.631, 0.843, 0.416)
colorWHO3 = [102, 179, 230] / 255;  % Blue (0.4, 0.702, 0.902)

% Figure creation function (headless-aware)
if headless
    create_fig = @(varargin) figure(varargin{:}, 'Visible', 'off');
else
    create_fig = @(varargin) figure(varargin{:});
end

if verbose
    fprintf('\n========================================================================\n');
    fprintf('  Explorative Datenanalyse (EDA)\n');
    fprintf('  Preprocessing: %s\n', pp_type);
    fprintf('========================================================================\n\n');
end

%% Load wavenumbers
load('data/wavenumbers.mat', 'wavenumbers_roi');
wavenumbers = wavenumbers_roi;

%% Step 1: Prepare data matrices
if verbose
    fprintf('Schritt 1: Daten vorbereiten...\n');
end

% Select preprocessing type
if strcmp(pp_type, 'PP1')
    spectra_col = 'CombinedSpectra_PP1';
    mean_col = 'MeanSpectrum_PP1';
else
    spectra_col = 'CombinedSpectra_PP2';
    mean_col = 'MeanSpectrum_PP2';
    % For PP2, wavenumbers are binned
    wavenumbers = wavenumbers(1:4:end);  % Binning factor 4
end

% Build spectrum-level data matrix
n_probes = height(dataset_men);
all_spectra = [];
all_who_grades = [];
all_probe_ids = [];
all_is_train = [];  % Track which spectra are from training set

for i = 1:n_probes
    spectra_matrix = dataset_men.(spectra_col){i};
    n_spectra_probe = size(spectra_matrix, 1);
    
    all_spectra = [all_spectra; spectra_matrix];
    all_who_grades = [all_who_grades; repmat(dataset_men.WHO_Grade(i), n_spectra_probe, 1)];
    all_probe_ids = [all_probe_ids; repmat(i, n_spectra_probe, 1)];
    
    % Track if this probe is in training set
    is_train_probe = train_indices(i);
    all_is_train = [all_is_train; repmat(is_train_probe, n_spectra_probe, 1)];
end

if verbose
    fprintf('  Gesamtspektren: %d\n', size(all_spectra, 1));
    fprintf('  WHO-1: %d Spektren\n', sum(all_who_grades == 'WHO-1'));
    fprintf('  WHO-2: %d Spektren\n', sum(all_who_grades == 'WHO-2'));
    fprintf('  WHO-3: %d Spektren\n', sum(all_who_grades == 'WHO-3'));
    fprintf('  Training set: %d Spektren (WHO-1 & WHO-3)\n', sum(all_is_train));
    fprintf('  Test set: %d Spektren\n', sum(~all_is_train));
end

%% Step 2: Mean spectra per WHO grade with SD bands (separate plots)
if verbose
    fprintf('\nSchritt 2: Mittlere Spektren pro WHO-Grad erstellen...\n');
end

create_fig('Position', [100 100 1200 900]);
t = tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

for grade_idx = 1:3
    grade_name = sprintf('WHO-%d', grade_idx);
    grade_mask = all_who_grades == grade_name;
    
    if sum(grade_mask) > 0
        grade_spectra = all_spectra(grade_mask, :);
        mean_spectrum = mean(grade_spectra, 1);
        std_spectrum = std(grade_spectra, 0, 1);
        
        % Select color
        if grade_idx == 1
            color = colorWHO1;
        elseif grade_idx == 2
            color = colorWHO2;
        else
            color = colorWHO3;
        end
        
        % Create subplot for this grade
        nexttile;
        
        % Plot with SD band
        fill([wavenumbers, fliplr(wavenumbers)], ...
             [mean_spectrum + std_spectrum, fliplr(mean_spectrum - std_spectrum)], ...
             color, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        hold on;
        plot(wavenumbers, mean_spectrum, 'Color', color, 'LineWidth', 2);
        hold off;
        
        ylabel('Absorption (a.u.)', 'FontSize', 12);
        title(sprintf('%s - Mittelwert (n=%d Spektren)', grade_name, sum(grade_mask)), 'FontSize', 12);
        set(gca, 'FontSize', 12);
        xlim([min(wavenumbers) max(wavenumbers)]);
        
        % Disable scientific notation on y-axis
        ax = gca;
        ax.YAxis.Exponent = 0;
        
        % Only add xlabel to bottom plot
        if grade_idx == 3
            xlabel('Wellenzahl (cm^{-1})', 'FontSize', 12);
        end
    end
end

title(t, 'Mittlere Spektren pro WHO-Grad mit Standardabweichung', 'FontSize', 14);

saveas(gcf, fullfile(output_dir, sprintf('01_mean_spectra_WHO_grades_%s.png', pp_type)));
if verbose, fprintf('  Gespeichert: 01_mean_spectra_WHO_grades_%s.png\n', pp_type); end

%% Step 3: Individual probe mean spectra (tiled layout, grouped by WHO grade)
if verbose
    fprintf('\nSchritt 3: Einzelne Proben-Spektren erstellen...\n');
end

% Group probes by WHO grade
for grade_idx = 1:3
    grade_name = sprintf('WHO-%d', grade_idx);
    grade_mask = dataset_men.WHO_Grade == grade_name;
    grade_indices = find(grade_mask);
    n_probes_grade = length(grade_indices);
    
    if verbose
        fprintf('  %s: %d Proben gefunden\n', grade_name, n_probes_grade);
    end
    
    if n_probes_grade == 0
        continue;
    end
    
    % Select color
    if grade_idx == 1
        color = colorWHO1;
    elseif grade_idx == 2
        color = colorWHO2;
    else
        color = colorWHO3;
    end
    
    % Create figures for this grade (15 probes per figure)
    probes_per_fig = 15;
    n_figures = ceil(n_probes_grade / probes_per_fig);
    
    if verbose
        fprintf('    Erstelle %d Figur(en) für %s\n', n_figures, grade_name);
    end
    
    for fig_idx = 1:n_figures
        create_fig('Position', [100 100 1400 900]);
        t = tiledlayout(5, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
        
        start_idx = (fig_idx - 1) * probes_per_fig + 1;
        end_idx = min(fig_idx * probes_per_fig, n_probes_grade);
        
        for local_idx = start_idx:end_idx
            probe_idx = grade_indices(local_idx);
            tile_idx = local_idx - start_idx + 1;
            nexttile(tile_idx);
            
            % Get probe data
            probe_spectra = dataset_men.(spectra_col){probe_idx};
            mean_spectrum = mean(probe_spectra, 1);
            std_spectrum = std(probe_spectra, 0, 1);
            diss_id = dataset_men.Diss_ID{probe_idx};
            
            % Plot
            fill([wavenumbers, fliplr(wavenumbers)], ...
                 [mean_spectrum + std_spectrum, fliplr(mean_spectrum - std_spectrum)], ...
                 color, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
            hold on;
            plot(wavenumbers, mean_spectrum, 'Color', color, 'LineWidth', 1.5);
            hold off;
            
            title(sprintf('%s', diss_id), 'FontSize', 10);
            if tile_idx > 12
                xlabel('Wellenzahl (cm^{-1})', 'FontSize', 9);
            end
            if mod(tile_idx-1, 3) == 0
                ylabel('Absorption (a.u.)', 'FontSize', 9);
            end
            set(gca, 'FontSize', 9);
            xlim([min(wavenumbers) max(wavenumbers)]);
            
            % Disable scientific notation on y-axis
            ax = gca;
            ax.YAxis.Exponent = 0;
        end
        
        if n_figures > 1
            title(t, sprintf('%s Proben-Spektren (Teil %d/%d, Proben %d-%d)', ...
                grade_name, fig_idx, n_figures, start_idx, end_idx), 'FontSize', 12);
        else
            title(t, sprintf('%s Proben-Spektren (n=%d)', grade_name, n_probes_grade), 'FontSize', 12);
        end
        
        saveas(gcf, fullfile(output_dir, sprintf('02_probe_spectra_%s_%s_part%d.png', ...
            pp_type, strrep(grade_name, '-', ''), fig_idx)));
        if verbose
            fprintf('  Gespeichert: 02_probe_spectra_%s_%s_part%d.png\n', ...
                pp_type, strrep(grade_name, '-', ''), fig_idx);
        end
    end
end

%% Step 4: Descriptive statistics per class and wavenumber
if verbose
    fprintf('\nSchritt 4: Deskriptive Statistik berechnen...\n');
end

stats = struct();
for grade_idx = 1:3  % All three WHO grades
    grade_name = sprintf('WHO-%d', grade_idx);
    grade_mask = all_who_grades == grade_name;
    grade_spectra = all_spectra(grade_mask, :);
    
    stats.(sprintf('WHO%d', grade_idx)).n_spectra = sum(grade_mask);
    stats.(sprintf('WHO%d', grade_idx)).mean = mean(grade_spectra, 1);
    stats.(sprintf('WHO%d', grade_idx)).std = std(grade_spectra, 0, 1);
    stats.(sprintf('WHO%d', grade_idx)).median = median(grade_spectra, 1);
    stats.(sprintf('WHO%d', grade_idx)).q25 = quantile(grade_spectra, 0.25, 1);
    stats.(sprintf('WHO%d', grade_idx)).q75 = quantile(grade_spectra, 0.75, 1);
end

%% Step 5: PCA for outlier detection (WHO-1 and WHO-3 only - training set)
if verbose
    fprintf('\nSchritt 5: PCA für Ausreißer-Erkennung durchführen...\n');
end

% Use only WHO-1 and WHO-3 spectra from TRAINING SET for PCA
pca_mask = all_is_train & ((all_who_grades == 'WHO-1') | (all_who_grades == 'WHO-3'));
X_pca = all_spectra(pca_mask, :);
who_grades_pca = all_who_grades(pca_mask);
probe_ids_pca = all_probe_ids(pca_mask);

if verbose
    fprintf('  PCA mit NUR TRAINING SET WHO-1 und WHO-3 Spektren\n');
    fprintf('  Anzahl Spektren für PCA: %d\n', size(X_pca, 1));
    fprintf('  WHO-1: %d Spektren\n', sum(who_grades_pca == 'WHO-1'));
    fprintf('  WHO-3: %d Spektren\n', sum(who_grades_pca == 'WHO-3'));
end

% Mean-center the data
X_mean = mean(X_pca, 1);
X_centered = X_pca - X_mean;

% Perform PCA on centered data
[coeff, score, latent, ~, explained] = pca(X_centered);

if verbose
    fprintf('  PCA abgeschlossen\n');
    fprintf('  PC1 erklärt %.2f%% der Varianz\n', explained(1));
    fprintf('  PC2 erklärt %.2f%% der Varianz\n', explained(2));
    fprintf('  PC1-3 zusammen: %.2f%%\n', sum(explained(1:3)));
end

% Calculate T² statistic (Hotelling's T²)
% Use fewer PCs for outlier detection to capture more variance in residuals
n_pcs = min(5, size(score, 2));  % Use first 5 PCs for outlier detection
T2 = zeros(size(score, 1), 1);
for i = 1:size(score, 1)
    T2(i) = sum((score(i, 1:n_pcs).^2) ./ latent(1:n_pcs)');
end

% Calculate Q statistic (SPE - Squared Prediction Error)
% Reconstruct data using only first n_pcs components
X_reconstructed = score(:, 1:n_pcs) * coeff(:, 1:n_pcs)';
% Calculate residuals in centered space
residuals = X_centered - X_reconstructed;
Q = sum(residuals.^2, 2);

if verbose
    fprintf('  Verwende %d PCs für Ausreißer-Erkennung\n', n_pcs);
    fprintf('  Q-Statistik: Min=%.4f, Max=%.4f, Mean=%.4f\n', min(Q), max(Q), mean(Q));
end

% Determine thresholds
T2_limit = t2_threshold * mean(T2) + t2_threshold * std(T2);
Q_limit = q_threshold * mean(Q) + q_threshold * std(Q);

% Identify outliers
outliers_T2 = T2 > T2_limit;
outliers_Q = Q > Q_limit;
outliers_both = outliers_T2 & outliers_Q;

if verbose
    fprintf('  T² Schwelle: %.2f (%.1f%% Ausreißer)\n', T2_limit, 100*sum(outliers_T2)/length(T2));
    fprintf('  Q Schwelle: %.2f (%.1f%% Ausreißer)\n', Q_limit, 100*sum(outliers_Q)/length(Q));
    fprintf('  Beide: %d Ausreißer (%.1f%%)\n', sum(outliers_both), 100*sum(outliers_both)/length(T2));
end

%% Step 6: T² vs Q plot
if verbose
    fprintf('\nSchritt 6: T²-Q-Diagramm erstellen...\n');
end

create_fig('Position', [100 100 900 700]);
hold on;

% Plot threshold lines
xline(T2_limit, '--r', 'LineWidth', 1.5, 'Label', sprintf('T² Schwelle (%.1f)', T2_limit), ...
      'LabelHorizontalAlignment', 'center', 'FontSize', 10);
yline(Q_limit, '--r', 'LineWidth', 1.5, 'Label', sprintf('Q Schwelle (%.1f)', Q_limit), ...
      'LabelVerticalAlignment', 'bottom', 'FontSize', 10);

% Plot points by WHO grade
for grade_idx = [1, 3]  % Only WHO-1 and WHO-3 (training set)
    grade_name = sprintf('WHO-%d', grade_idx);
    grade_mask = who_grades_pca == grade_name;
    
    if grade_idx == 1
        color = colorWHO1;
    else
        color = colorWHO3;
    end
    
    scatter(T2(grade_mask), Q(grade_mask), 50, color, 'filled', 'MarkerFaceAlpha', 0.6, ...
            'DisplayName', grade_name);
end

% Highlight outliers
scatter(T2(outliers_both), Q(outliers_both), 100, 'k', 'x', 'LineWidth', 2, ...
        'DisplayName', 'Beide Ausreißer');

hold off;
xlabel('T² Statistik', 'FontSize', 12);
ylabel('Q Statistik (SPE)', 'FontSize', 12);
title('T²-Q-Diagramm zur Ausreißer-Erkennung', 'FontSize', 12);
legend('Location', 'best', 'FontSize', 10);
set(gca, 'FontSize', 12);
grid on;

saveas(gcf, fullfile(output_dir, sprintf('03_T2_Q_plot_%s.png', pp_type)));
if verbose, fprintf('  Gespeichert: 03_T2_Q_plot_%s.png\n', pp_type); end

%% Step 7: PCA score plots
if verbose
    fprintf('\nSchritt 7: PCA-Score-Diagramme erstellen...\n');
end

% PC1 vs PC2
create_fig('Position', [100 100 1200 500]);
subplot(1, 2, 1);
hold on;
for grade_idx = [1, 3]  % Only WHO-1 and WHO-3 (training set)
    grade_name = sprintf('WHO-%d', grade_idx);
    grade_mask = who_grades_pca == grade_name;
    
    if grade_idx == 1
        color = colorWHO1;
    else
        color = colorWHO3;
    end
    
    scatter(score(grade_mask, 1), score(grade_mask, 2), 50, color, 'filled', ...
            'MarkerFaceAlpha', 0.6, 'DisplayName', grade_name);
end
scatter(score(outliers_both, 1), score(outliers_both, 2), 100, 'k', 'x', 'LineWidth', 2, ...
        'DisplayName', 'Ausreißer (T²+Q)');
hold off;
xlabel(sprintf('PC1 (%.1f%%)', explained(1)), 'FontSize', 12);
ylabel(sprintf('PC2 (%.1f%%)', explained(2)), 'FontSize', 12);
title('PCA Score Plot: PC1 vs PC2', 'FontSize', 12);
legend('Location', 'best');
set(gca, 'FontSize', 12);
grid on;

% PC1 vs PC3
subplot(1, 2, 2);
hold on;
for grade_idx = [1, 3]  % Only WHO-1 and WHO-3 (training set)
    grade_name = sprintf('WHO-%d', grade_idx);
    grade_mask = who_grades_pca == grade_name;
    
    if grade_idx == 1
        color = colorWHO1;
    else
        color = colorWHO3;
    end
    
    scatter(score(grade_mask, 1), score(grade_mask, 3), 50, color, 'filled', ...
            'MarkerFaceAlpha', 0.6, 'DisplayName', grade_name);
end
scatter(score(outliers_both, 1), score(outliers_both, 3), 100, 'k', 'x', 'LineWidth', 2, ...
        'DisplayName', 'Ausreißer (T²+Q)');
hold off;
xlabel(sprintf('PC1 (%.1f%%)', explained(1)), 'FontSize', 12);
ylabel(sprintf('PC3 (%.1f%%)', explained(3)), 'FontSize', 12);
title('PCA Score Plot: PC1 vs PC3', 'FontSize', 12);
legend('Location', 'best');
set(gca, 'FontSize', 12);
grid on;

saveas(gcf, fullfile(output_dir, sprintf('04_PCA_scores_%s.png', pp_type)));
if verbose, fprintf('  Gespeichert: 04_PCA_scores_%s.png\n', pp_type); end

%% Step 8: Scree plot (Cumulative Variance Explained)
if verbose
    fprintf('\nSchritt 8: Scree-Plot erstellen...\n');
end

% Determine number of PCs needed for 95% variance
cumsum_explained = cumsum(explained);
n_pcs_95 = find(cumsum_explained >= 95, 1, 'first');

% Create scree plot
n_pcs_to_plot = min(100, length(explained));  % Plot first 100 PCs or all if fewer
create_fig('Position', [100 100 1000 600]);

% Individual variance
yyaxis left
bar(1:n_pcs_to_plot, explained(1:n_pcs_to_plot), 'FaceColor', [0.3 0.3 0.8], 'EdgeColor', 'none');
ylabel('Varianz erklärt (%)', 'FontSize', 12);
ylim([0 max(explained(1:n_pcs_to_plot))*1.1]);
set(gca, 'YColor', [0.3 0.3 0.8]);

% Cumulative variance
yyaxis right
plot(1:n_pcs_to_plot, cumsum_explained(1:n_pcs_to_plot), 'r-', 'LineWidth', 2);
hold on;
% Mark 95% threshold
yline(95, '--k', 'LineWidth', 1.5, 'Label', '95%', 'LabelHorizontalAlignment', 'left');
% Mark PC achieving 95%
plot(n_pcs_95, cumsum_explained(n_pcs_95), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
text(n_pcs_95, cumsum_explained(n_pcs_95) - 5, sprintf('PC%d', n_pcs_95), ...
     'FontSize', 10, 'HorizontalAlignment', 'center');
hold off;
ylabel('Kumulative Varianz (%)', 'FontSize', 12);
ylim([0 105]);
set(gca, 'YColor', [0.8 0.1 0.1]);

xlabel('Hauptkomponente', 'FontSize', 12);
title('Scree Plot: Varianz erklärt durch Hauptkomponenten', 'FontSize', 14);
set(gca, 'FontSize', 12);
grid on;

if verbose
    fprintf('  %d PCs erklären 95%% der Varianz (kumulativ: %.2f%%)\n', ...
        n_pcs_95, cumsum_explained(n_pcs_95));
end

saveas(gcf, fullfile(output_dir, sprintf('05_scree_plot_%s.png', pp_type)));
if verbose, fprintf('  Gespeichert: 05_scree_plot_%s.png\n', pp_type); end

%% Step 9: Loadings plots
if verbose
    fprintf('\nSchritt 9: Loadings-Diagramme erstellen...\n');
end

if verbose
    fprintf('  %d PCs für Loadings-Plot\n', n_pcs_95);
end

% Create figure with appropriate layout
n_rows = ceil(n_pcs_95 / 3);
n_cols = min(3, n_pcs_95);

create_fig('Position', [100 100 1200 400*n_rows]);
for pc_idx = 1:n_pcs_95
    subplot(n_rows, n_cols, pc_idx);
    plot(wavenumbers, coeff(:, pc_idx), 'LineWidth', 1.5, 'Color', [0.2 0.2 0.2]);
    xlabel('Wellenzahl (cm^{-1})', 'FontSize', 11);
    ylabel('Loading', 'FontSize', 11);
    title(sprintf('PC%d (%.1f%%, Kum: %.1f%%)', pc_idx, explained(pc_idx), cumsum_explained(pc_idx)), ...
          'FontSize', 11);
    set(gca, 'FontSize', 10);
    xlim([min(wavenumbers) max(wavenumbers)]);
    grid on;
end

saveas(gcf, fullfile(output_dir, sprintf('06_PCA_loadings_95percent_%s.png', pp_type)));
if verbose, fprintf('  Gespeichert: 06_PCA_loadings_95percent_%s.png\n', pp_type); end

%% Step 9b: First 15 PCs plot (detailed view)
if verbose
    fprintf('\nSchritt 9b: Erste 15 PCs im Detail...\n');
end

n_pcs_detail = min(15, length(explained));
create_fig('Position', [100 100 1400 1000]);
for pc_idx = 1:n_pcs_detail
    subplot(5, 3, pc_idx);
    plot(wavenumbers, coeff(:, pc_idx), 'LineWidth', 1.5, 'Color', [0.2 0.2 0.2]);
    xlabel('Wellenzahl (cm^{-1})', 'FontSize', 10);
    ylabel('Loading', 'FontSize', 10);
    title(sprintf('PC%d (%.1f%%, Kum: %.1f%%)', pc_idx, explained(pc_idx), cumsum_explained(pc_idx)), ...
          'FontSize', 10);
    set(gca, 'FontSize', 9);
    xlim([min(wavenumbers) max(wavenumbers)]);
    grid on;
end

saveas(gcf, fullfile(output_dir, sprintf('06b_PCA_loadings_first15_%s.png', pp_type)));
if verbose, fprintf('  Gespeichert: 06b_PCA_loadings_first15_%s.png\n', pp_type); end

%% Step 10: Combined outlier spectra plot
if verbose
    fprintf('\nSchritt 10: Kombinierte Ausreißer-Spektren erstellen...\n');
end

% Identify different types of outliers
high_Q_only = outliers_Q & ~outliers_T2;
high_T2_only = outliers_T2 & ~outliers_Q;
both_outliers = outliers_both;

if verbose
    fprintf('  Nur hohe Q-Statistik: %d Spektren\n', sum(high_Q_only));
    fprintf('  Nur hohe T²-Statistik: %d Spektren\n', sum(high_T2_only));
    fprintf('  Beide (T² + Q): %d Spektren\n', sum(both_outliers));
end

% Calculate mean of non-outlier spectra for reference
all_outliers = outliers_Q | outliers_T2;
non_outliers = ~all_outliers;
mean_non_outliers = mean(X_pca(non_outliers, :), 1);

if verbose
    fprintf('  Nicht-Ausreißer Spektren: %d (für Mittelwert-Referenz)\n', sum(non_outliers));
end

% Determine global y-axis limits for uniform scaling
all_data = X_pca;
y_min_global = min(all_data(:));
y_max_global = max(all_data(:));
y_range = y_max_global - y_min_global;
y_limits = [y_min_global - 0.05*y_range, y_max_global + 0.05*y_range];

% Create combined plot
create_fig('Position', [100 100 1400 800]);

% Plot 1: High Q outliers only
subplot(3, 1, 1);
hold on;
% Plot mean of non-outliers as reference
plot(wavenumbers, mean_non_outliers, 'k-', 'LineWidth', 2, 'DisplayName', 'Mittelwert Nicht-Ausreißer');
% Plot outliers
if sum(high_Q_only) > 0
    plot(wavenumbers, X_pca(high_Q_only, :)', 'Color', [0.8 0.3 0.3, 0.3], 'LineWidth', 0.5);
end
hold off;
ylabel('Absorption (a.u.)', 'FontSize', 11);
title(sprintf('Hohe Q-Statistik Ausreißer (n=%d)', sum(high_Q_only)), 'FontSize', 12);
set(gca, 'FontSize', 11);
xlim([min(wavenumbers) max(wavenumbers)]);
ylim(y_limits);
% Disable scientific notation on y-axis
ax = gca;
ax.YAxis.Exponent = 0;
grid on;

% Plot 2: High T² outliers only
subplot(3, 1, 2);
hold on;
% Plot mean of non-outliers as reference
plot(wavenumbers, mean_non_outliers, 'k-', 'LineWidth', 2, 'DisplayName', 'Mittelwert Nicht-Ausreißer');
% Plot outliers
if sum(high_T2_only) > 0
    plot(wavenumbers, X_pca(high_T2_only, :)', 'Color', [0.3 0.3 0.8, 0.3], 'LineWidth', 0.5);
end
hold off;
ylabel('Absorption (a.u.)', 'FontSize', 11);
title(sprintf('Hohe T²-Statistik Ausreißer (n=%d)', sum(high_T2_only)), 'FontSize', 12);
set(gca, 'FontSize', 11);
xlim([min(wavenumbers) max(wavenumbers)]);
ylim(y_limits);
% Disable scientific notation on y-axis
ax = gca;
ax.YAxis.Exponent = 0;
grid on;

% Plot 3: Both outliers
subplot(3, 1, 3);
hold on;
% Plot mean of non-outliers as reference
plot(wavenumbers, mean_non_outliers, 'k-', 'LineWidth', 2, 'DisplayName', 'Mittelwert Nicht-Ausreißer');
% Plot outliers
if sum(both_outliers) > 0
    plot(wavenumbers, X_pca(both_outliers, :)', 'Color', [0.3 0.7 0.3, 0.3], 'LineWidth', 0.5);
end
hold off;
xlabel('Wellenzahl (cm^{-1})', 'FontSize', 11);
ylabel('Absorption (a.u.)', 'FontSize', 11);
title(sprintf('Beide Ausreißer (T² + Q) (n=%d)', sum(both_outliers)), 'FontSize', 12);
set(gca, 'FontSize', 11);
xlim([min(wavenumbers) max(wavenumbers)]);
ylim(y_limits);
% Disable scientific notation on y-axis
ax = gca;
ax.YAxis.Exponent = 0;
grid on;

saveas(gcf, fullfile(output_dir, sprintf('07_combined_outliers_%s.png', pp_type)));
if verbose, fprintf('  Gespeichert: 07_combined_outliers_%s.png\n', pp_type); end

%% Step 11: Outlier spectra visualization
if verbose
    fprintf('\nSchritt 11: Ausreißer-Spektren visualisieren...\n');
end

% Select representative outliers
high_Q_idx = find(outliers_Q & ~outliers_T2);
high_T2_idx = find(outliers_T2 & ~outliers_Q);
both_idx = find(outliers_both);

n_examples = min([3, length(high_Q_idx), length(high_T2_idx), length(both_idx)]);

if n_examples > 0
    create_fig('Position', [100 100 1400 900]);
    t = tiledlayout(3, n_examples, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    % Plot high Q outliers
    for i = 1:n_examples
        if i <= length(high_Q_idx)
            idx = high_Q_idx(i);
            nexttile;
            plot_outlier_spectrum(X_pca(idx, :), who_grades_pca(idx), ...
                                  stats, wavenumbers, colorWHO1, colorWHO2, colorWHO3);
            title(sprintf('Hohe Q: Spektrum %d', idx), 'FontSize', 10);
            if i == 1, ylabel('Hohe Q-Statistik', 'FontSize', 11); end
        end
    end
    
    % Plot high T² outliers
    for i = 1:n_examples
        if i <= length(high_T2_idx)
            idx = high_T2_idx(i);
            nexttile;
            plot_outlier_spectrum(X_pca(idx, :), who_grades_pca(idx), ...
                                  stats, wavenumbers, colorWHO1, colorWHO2, colorWHO3);
            title(sprintf('Hohe T²: Spektrum %d', idx), 'FontSize', 10);
            if i == 1, ylabel('Hohe T²-Statistik', 'FontSize', 11); end
        end
    end
    
    % Plot both outliers
    for i = 1:n_examples
        if i <= length(both_idx)
            idx = both_idx(i);
            nexttile;
            plot_outlier_spectrum(X_pca(idx, :), who_grades_pca(idx), ...
                                  stats, wavenumbers, colorWHO1, colorWHO2, colorWHO3);
            title(sprintf('Beide: Spektrum %d', idx), 'FontSize', 10);
            if i == 1, ylabel('Hohe T²+Q', 'FontSize', 11); end
        end
    end
    
    title(t, 'Ausreißer-Spektren im Vergleich zum Klassenmittelwert', 'FontSize', 12);
    
    saveas(gcf, fullfile(output_dir, sprintf('08_outlier_spectra_examples_%s.png', pp_type)));
    if verbose, fprintf('  Gespeichert: 08_outlier_spectra_examples_%s.png\n', pp_type); end
end

%% Save results
eda_results = struct();
eda_results.preprocessing_type = pp_type;
eda_results.n_spectra = size(all_spectra, 1);
eda_results.n_probes = n_probes;
eda_results.wavenumbers = wavenumbers;
eda_results.stats = stats;
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

% Save additional information needed for downstream pipeline
eda_results.probe_ids_pca = probe_ids_pca;  % Probe IDs for PCA spectra
eda_results.is_train = all_is_train;  % Which spectra are from training set
eda_results.X_mean = X_mean;  % Mean spectrum used for PCA centering
eda_results.wavenumbers = wavenumbers;  % Wavenumber vector
eda_results.n_pcs_used = n_pcs;  % Number of PCs used for outlier detection

save(fullfile(output_dir, sprintf('eda_results_%s.mat', pp_type)), 'eda_results');

if verbose
    fprintf('\n========================================================================\n');
    fprintf('  EDA abgeschlossen!\n');
    fprintf('  Ergebnisse gespeichert in: %s\n', output_dir);
    fprintf('  Outlier-gefilterte Daten bereit für Pipeline!\n');
    fprintf('========================================================================\n\n');
end

end

%% Helper function: Plot outlier spectrum
function plot_outlier_spectrum(spectrum, who_grade, stats, wavenumbers, colorWHO1, colorWHO2, colorWHO3)
    if who_grade == 'WHO-1'
        class_mean = stats.WHO1.mean;
        class_std = stats.WHO1.std;
        color = colorWHO1;
        grade_str = 'WHO-1';
    elseif who_grade == 'WHO-2'
        class_mean = stats.WHO2.mean;
        class_std = stats.WHO2.std;
        color = colorWHO2;
        grade_str = 'WHO-2';
    else
        class_mean = stats.WHO3.mean;
        class_std = stats.WHO3.std;
        color = colorWHO3;
        grade_str = 'WHO-3';
    end
    
    % Plot class mean with SD band
    fill([wavenumbers, fliplr(wavenumbers)], ...
         [class_mean + class_std, fliplr(class_mean - class_std)], ...
         color, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    hold on;
    plot(wavenumbers, class_mean, 'Color', color, 'LineWidth', 1.5, ...
         'DisplayName', sprintf('%s Mittelwert', grade_str));
    plot(wavenumbers, spectrum, 'k-', 'LineWidth', 1, 'DisplayName', 'Ausreißer');
    hold off;
    
    xlabel('Wellenzahl (cm^{-1})', 'FontSize', 9);
    ylabel('Absorption (a.u.)', 'FontSize', 9);
    set(gca, 'FontSize', 9);
    xlim([min(wavenumbers) max(wavenumbers)]);
    % Disable scientific notation on y-axis
    ax = gca;
    ax.YAxis.Exponent = 0;
    legend('Location', 'best', 'FontSize', 8);
end
