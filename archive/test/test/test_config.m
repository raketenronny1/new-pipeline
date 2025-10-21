function cfg = test_config()
    % Create a configuration for test runs
    
    % Get base configuration
    cfg = config();
    
    % Get test directory path
    test_dir = fileparts(mfilename('fullpath'));
    
    % Override paths for test environment with absolute paths
    cfg.paths.data = fullfile(test_dir, 'data');
    cfg.paths.results = fullfile(test_dir, 'results');
    cfg.paths.models = fullfile(test_dir, 'models');
    cfg.paths.qc = fullfile(test_dir, 'results', 'qc');
    
    % Ensure directories exist
    if ~exist(cfg.paths.data, 'dir')
        mkdir(cfg.paths.data);
    end
    if ~exist(cfg.paths.results, 'dir')
        mkdir(cfg.paths.results);
    end
    if ~exist(cfg.paths.models, 'dir')
        mkdir(cfg.paths.models);
    end
    if ~exist(cfg.paths.qc, 'dir')
        mkdir(cfg.paths.qc);
    end
    
    % Test-specific parameters
    cfg.test = struct();
    cfg.test.n_samples = 40;        % Total number of test samples
    cfg.test.train_ratio = 0.7;     % Ratio of training samples
    
    % Reduced CV parameters for faster testing
    cfg.cv.n_folds = 3;            % Reduced from 5
    cfg.cv.n_repeats = 10;         % Reduced from 50
    
    % Adjust QC thresholds for real test data
    cfg.qc.snr_threshold = 5;           % Reduced from 10, but not too low
    cfg.qc.max_absorbance = 2.0;        % Increased slightly from 1.8
    cfg.qc.baseline_sd_threshold = 0.05; % More permissive than 0.02
    cfg.qc.amide_ratio_min = 0.8;       % Reduced from 1.2, but still meaningful
    cfg.qc.amide_ratio_max = 4.0;       % Increased from 3.5, but not too high
    cfg.qc.min_spectra_per_sample = 3;  % Reduced from 100 for small test data
    cfg.qc.within_sample_corr_threshold = 0.7;  % Reduced from 0.85 but still requires correlation
    cfg.qc.outlier_confidence = 0.95;    % Slightly reduced to keep more samples
    cfg.qc.max_samples_to_exclude = 1;   % Reduced to keep more samples
    
    % Set random seed for reproducibility
    cfg.random_seed = 42;
end