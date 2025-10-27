%% RUN_EDA - Run exploratory data analysis with outlier detection
%
% Performs comprehensive exploratory data analysis on FTIR training data including:
%   - PCA analysis (5 components for visualization, 15+ for classification)
%   - T² and Q statistic outlier detection
%   - Visualization of data structure and quality
%   - Saves results for downstream pipeline use
%
% SYNTAX:
%   run_eda()
%   run_eda('Name', Value, ...)
%   eda_results = run_eda(...)
%
% OPTIONAL NAME-VALUE PAIRS:
%   'PreprocessingType' - Preprocessing method: 'PP1', 'PP2', etc. (default: 'PP1')
%   'Verbose'           - Display detailed output (default: true)
%   'CreatePlots'       - Generate visualization plots (default: true)
%   'TrainDataFile'     - Path to training data file (default: 'data/data_table_train.mat')
%
% OUTPUTS:
%   eda_results - Structure containing:
%                 * pca: PCA model and outlier flags
%                 * X_mean: Mean spectrum for centering
%                 * wavenumbers: Wavenumber values
%                 * probe_ids_pca: ProbeUID mapping
%                 * is_train: Training set indicator
%                 Saved to: results/eda/eda_results_PP1.mat
%
% PREPROCESSING METHODS:
%   'PP1' - Standard normalization (recommended)
%   'PP2' - Alternative preprocessing
%   (See exploratory_data_analysis.m for details)
%
% EXAMPLES:
%   % Run with default settings (recommended)
%   run_eda()
%
%   % Run without plots (for batch processing)
%   run_eda('CreatePlots', false)
%
%   % Use alternative preprocessing
%   run_eda('PreprocessingType', 'PP2')
%
%   % Capture results
%   results = run_eda('Verbose', true);
%
% OUTPUTS CREATED:
%   results/eda/eda_results_PP1.mat  - Main results file
%   results/eda/*.png                - Visualization plots (if CreatePlots=true)
%
% NOTES:
%   - Requires data_table_train.mat to exist
%   - Uses only training data (test set not analyzed to prevent leakage)
%   - Results are used by load_pipeline_data.m for outlier filtering
%   - PCA model (15 components) used by LDA classifier
%
% See also: exploratory_data_analysis, load_pipeline_data, run_pipeline

function eda_results = run_eda(varargin)
    %% Parse input arguments
    p = inputParser;
    addParameter(p, 'PreprocessingType', 'PP1', @ischar);
    addParameter(p, 'Verbose', true, @islogical);
    addParameter(p, 'CreatePlots', true, @islogical);
    addParameter(p, 'TrainDataFile', 'data/data_table_train.mat', @ischar);
    parse(p, varargin{:});
    
    opts = p.Results;
    
    %% Display header
    if opts.Verbose
        fprintf('\n');
        fprintf('═══════════════════════════════════════════════════════════\n');
        fprintf(' EXPLORATORY DATA ANALYSIS (EDA)\n');
        fprintf('═══════════════════════════════════════════════════════════\n');
        fprintf('Preprocessing: %s\n', opts.PreprocessingType);
        fprintf('Create plots: %s\n', mat2str(opts.CreatePlots));
        fprintf('\n');
    end
    
    %% Check training data exists
    if ~exist(opts.TrainDataFile, 'file')
        error('run_eda:FileNotFound', ...
              ['Training data file not found: %s\n' ...
               'Please run split_train_test first to generate training data.'], ...
              opts.TrainDataFile);
    end
    
    %% Load training data
    if opts.Verbose
        fprintf('Loading training data...\n');
    end
    
    m_train = matfile(opts.TrainDataFile);
    data_table_train = m_train.data_table_train;
    
    if opts.Verbose
        fprintf('  Training set: %d probes (WHO-1 & WHO-3)\n', height(data_table_train));
    end
    
    % For EDA, we only use training data
    dataset_men = data_table_train;
    train_indices = true(height(dataset_men), 1);  % All are training samples
    
    if opts.Verbose
        fprintf('  Dataset ready: %d probes total\n', height(dataset_men));
        fprintf('  All marked as training for PCA\n\n');
    end
    
    %% Run EDA
    if opts.Verbose
        fprintf('Starting EDA analysis...\n');
        fprintf('This may take several minutes for large datasets...\n\n');
    end
    
    tic;
    
    % Choose function based on plot option
    if opts.CreatePlots
        eda_results = exploratory_data_analysis(dataset_men, ...
            'PreprocessingType', opts.PreprocessingType, ...
            'Verbose', opts.Verbose, ...
            'TrainIndices', train_indices);
    else
        % Use no-plots version for faster processing
        eda_results = exploratory_data_analysis_no_plots(dataset_men, ...
            'PreprocessingType', opts.PreprocessingType, ...
            'Verbose', opts.Verbose, ...
            'TrainIndices', train_indices);
    end
    
    elapsed = toc;
    
    %% Summary
    if opts.Verbose
        fprintf('\n');
        fprintf('═══════════════════════════════════════════════════════════\n');
        fprintf(' EDA SUMMARY\n');
        fprintf('═══════════════════════════════════════════════════════════\n');
        fprintf('Total time: %.1f seconds (%.1f minutes)\n', elapsed, elapsed/60);
        fprintf('Outliers detected: %d / %d spectra (%.1f%%)\n', ...
                sum(eda_results.pca.outliers_both), ...
                length(eda_results.pca.outliers_both), ...
                100*sum(eda_results.pca.outliers_both)/length(eda_results.pca.outliers_both));
        fprintf('PCA components: %d (%.1f%% variance)\n', ...
                size(eda_results.pca.coeff, 2), ...
                sum(eda_results.pca.explained(1:min(15, end))));
        
        fprintf('\nResults saved to: results/eda/\n');
        
        if opts.CreatePlots
            % List generated plots
            plot_files = dir('results/eda/*.png');
            if ~isempty(plot_files)
                fprintf('\nGenerated %d visualization plots:\n', length(plot_files));
                for i = 1:min(5, length(plot_files))
                    fprintf('  %2d. %s\n', i, plot_files(i).name);
                end
                if length(plot_files) > 5
                    fprintf('  ... and %d more\n', length(plot_files) - 5);
                end
            end
        end
        
        fprintf('\n✓ EDA completed successfully!\n');
        fprintf('═══════════════════════════════════════════════════════════\n\n');
    end
end
