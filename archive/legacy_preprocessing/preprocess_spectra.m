function [processed_spectra, processed_wavenumbers] = preprocess_spectra(spectra, wavenumbers, config)
% PREPROCESS_SPECTRA Main orchestrator for spectral preprocessing
%
% SYNTAX:
%   [processed_spectra, processed_wavenumbers] = preprocess_spectra(spectra, wavenumbers, config)
%
% INPUTS:
%   spectra     - Matrix of spectra (rows = spectra, cols = wavenumbers)
%                 Can also be a single row vector
%   wavenumbers - Row vector of wavenumber values
%   config      - Configuration structure from create_preprocessing_config
%                 OR string: 'PP1', 'PP2' (will create config automatically)
%
% OUTPUTS:
%   processed_spectra     - Preprocessed spectra matrix
%   processed_wavenumbers - Wavenumber vector (may be binned if configured)
%
% DESCRIPTION:
%   Applies a sequence of preprocessing steps to FTIR spectra based on the
%   provided configuration. This is the main entry point for all spectral
%   preprocessing in the meningioma FTIR pipeline.
%   
%   Processing order (BSNC):
%   1. Binning (B) - if enabled
%   2. Smoothing (S) - if enabled
%   3. Normalization (N) - if enabled
%   4. baseline Correction (C) via derivative - if enabled
%   
%   The function handles both PP1 and PP2 preprocessing approaches:
%   
%   PP1 (Standard, default for analysis):
%   - Skip binning
%   - Skip smoothing
%   - Apply vector normalization (L2)
%   - Apply 2nd derivative (SG: window=5, poly=2)
%   
%   PP2 (Enhanced with noise reduction):
%   - Apply binning (factor 4)
%   - Apply SG smoothing (window=11, poly=2, deriv=0)
%   - Apply vector normalization (L2)
%   - Apply 2nd derivative (SG: window=5, poly=2)
%
% EXAMPLE:
%   % Using string shorthand
%   [processed, wn] = preprocess_spectra(raw_spectra, wavenumbers, 'PP1');
%   
%   % Using explicit config
%   cfg = create_preprocessing_config('PP2');
%   [processed, wn] = preprocess_spectra(raw_spectra, wavenumbers, cfg);
%   
%   % Custom configuration
%   cfg = create_preprocessing_config('custom');
%   cfg.apply_binning = true;
%   cfg.bin_factor = 2;
%   [processed, wn] = preprocess_spectra(raw_spectra, wavenumbers, cfg);
%
% See also: CREATE_PREPROCESSING_CONFIG, APPLY_BINNING, APPLY_VECTOR_NORMALIZATION
%
% Author: GitHub Copilot
% Date: 2025-10-24

%% Input validation
if nargin < 2
    error('preprocess_spectra:MissingInput', ...
        'At least spectra and wavenumbers are required.');
end

if nargin < 3 || isempty(config)
    config = 'PP1';  % Default to standard pipeline
end

% If config is a string, create the configuration
if ischar(config) || isstring(config)
    approach_name = char(config);
    config = create_preprocessing_config(approach_name);
end

% Validate inputs
if ~isnumeric(spectra) || ~ismatrix(spectra)
    error('preprocess_spectra:InvalidInput', 'Spectra must be a numeric matrix.');
end

if ~isvector(wavenumbers)
    error('preprocess_spectra:InvalidInput', 'Wavenumbers must be a vector.');
end

% Ensure wavenumbers is a row vector
if iscolumn(wavenumbers)
    wavenumbers = wavenumbers';
end

% Check dimensions match
if size(spectra, 2) ~= length(wavenumbers)
    error('preprocess_spectra:DimensionMismatch', ...
        'Number of columns in spectra (%d) must match wavenumber vector length (%d).', ...
        size(spectra, 2), length(wavenumbers));
end

%% Display preprocessing plan
if ~isempty(config.name)
    fprintf('\n=== Preprocessing with %s ===\n', config.name);
    fprintf('Description: %s\n', config.description);
    fprintf('Input: %d spectra x %d wavenumbers\n', size(spectra, 1), size(spectra, 2));
end

%% Initialize
processed_spectra = spectra;
processed_wavenumbers = wavenumbers;
step_num = 1;

%% Step 1: Binning (if enabled)
if config.apply_binning && config.bin_factor > 1
    fprintf('  Step %d: Binning (factor=%d)... ', step_num, config.bin_factor);
    tic;
    
    [processed_spectra, processed_wavenumbers] = apply_binning(...
        processed_spectra, processed_wavenumbers, config.bin_factor);
    
    fprintf('Done (%.2f s). New size: %d wavenumbers\n', toc, length(processed_wavenumbers));
    step_num = step_num + 1;
else
    fprintf('  Step %d: Binning - Skipped\n', step_num);
    step_num = step_num + 1;
end

%% Step 2: Smoothing (if enabled)
if config.apply_smoothing
    fprintf('  Step %d: SG Smoothing (window=%d, poly=%d)... ', ...
        step_num, config.smooth_window, config.smooth_poly_order);
    tic;
    
    processed_spectra = apply_sg_smoothing(...
        processed_spectra, config.smooth_window, config.smooth_poly_order);
    
    fprintf('Done (%.2f s)\n', toc);
    step_num = step_num + 1;
else
    fprintf('  Step %d: Smoothing - Skipped\n', step_num);
    step_num = step_num + 1;
end

%% Step 3: Normalization (if enabled)
if config.apply_normalization
    fprintf('  Step %d: %s Normalization... ', step_num, config.norm_type);
    tic;
    
    switch config.norm_type
        case 'L2'
            processed_spectra = apply_vector_normalization(processed_spectra);
            
        case 'SNV'
            % SNV normalization: (spectrum - mean) / std
            warning('preprocess_spectra:NotImplemented', ...
                'SNV normalization not yet implemented. Using L2 instead.');
            processed_spectra = apply_vector_normalization(processed_spectra);
            
        case 'MinMax'
            % Min-Max normalization: (spectrum - min) / (max - min)
            warning('preprocess_spectra:NotImplemented', ...
                'MinMax normalization not yet implemented. Using L2 instead.');
            processed_spectra = apply_vector_normalization(processed_spectra);
            
        case 'None'
            % Skip normalization
            fprintf('(None specified)');
            
        otherwise
            warning('preprocess_spectra:UnknownNormType', ...
                'Unknown normalization type "%s". Skipping normalization.', config.norm_type);
    end
    
    fprintf('Done (%.2f s)\n', toc);
    step_num = step_num + 1;
else
    fprintf('  Step %d: Normalization - Skipped\n', step_num);
    step_num = step_num + 1;
end

%% Step 4: Derivative / Baseline Correction (if enabled)
if config.apply_derivative && config.deriv_order > 0
    deriv_name = 'derivative';
    if config.deriv_order == 1
        deriv_name = '1st derivative';
    elseif config.deriv_order == 2
        deriv_name = '2nd derivative';
    end
    
    fprintf('  Step %d: %s (SG: window=%d, poly=%d)... ', ...
        step_num, deriv_name, config.deriv_window, config.deriv_poly_order);
    tic;
    
    processed_spectra = apply_sg_derivative(...
        processed_spectra, config.deriv_window, config.deriv_poly_order, config.deriv_order);
    
    fprintf('Done (%.2f s)\n', toc);
    step_num = step_num + 1;
else
    fprintf('  Step %d: Derivative - Skipped\n', step_num);
    step_num = step_num + 1;
end

%% Final quality check
num_nan = sum(isnan(processed_spectra(:)));
num_inf = sum(isinf(processed_spectra(:)));

if num_nan > 0 || num_inf > 0
    warning('preprocess_spectra:QualityIssue', ...
        'Output contains %d NaN and %d Inf values (%.2f%% of data).', ...
        num_nan, num_inf, 100 * (num_nan + num_inf) / numel(processed_spectra));
end

%% Summary
fprintf('=== Preprocessing Complete ===\n');
fprintf('Output: %d spectra x %d wavenumbers\n', ...
    size(processed_spectra, 1), size(processed_spectra, 2));
fprintf('Data quality: %.2f%% valid values\n\n', ...
    100 * sum(isfinite(processed_spectra(:))) / numel(processed_spectra));

end
