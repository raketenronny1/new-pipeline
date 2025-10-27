function smoothed_spectra = apply_sg_smoothing(spectra, window_size, polynomial_order)
% APPLY_SG_SMOOTHING Apply Savitzky-Golay smoothing filter (derivative order = 0)
%
% SYNTAX:
%   smoothed_spectra = apply_sg_smoothing(spectra, window_size, polynomial_order)
%
% INPUTS:
%   spectra           - Matrix of spectra (rows = spectra, cols = wavenumbers)
%                       Can also be a single row vector
%   window_size       - Filter window size (must be odd, default: 11)
%   polynomial_order  - Polynomial order for fitting (default: 2)
%
% OUTPUTS:
%   smoothed_spectra - Smoothed spectra (same dimensions as input)
%
% DESCRIPTION:
%   Applies Savitzky-Golay smoothing filter to reduce high-frequency noise
%   while preserving peak shapes. This is smoothing only (derivative order = 0),
%   NOT differentiation.
%   
%   The filter fits a polynomial to a sliding window of data points and
%   replaces the center point with the polynomial value.
%   
%   Recommended parameters for brain tissue FTIR:
%   - Window size: 11 points (can use 5-15 depending on noise level)
%   - Polynomial order: 2 (quadratic fit)
%   
%   Constraints:
%   - window_size must be odd
%   - window_size > polynomial_order
%   - window_size <= number of data points
%
% EDGE CASES:
%   - Spectra shorter than window: Uses original spectrum with warning
%   - NaN/Inf values: Attempts to filter, may propagate issues
%
% EXAMPLE:
%   % Smooth spectra with default parameters
%   smoothed = apply_sg_smoothing(my_spectra);
%   
%   % Custom parameters
%   smoothed = apply_sg_smoothing(my_spectra, 7, 2);
%
% See also: APPLY_SG_DERIVATIVE, SGOLAYFILT, PREPROCESS_SPECTRA
%
% Author: GitHub Copilot
% Date: 2025-10-24

%% Input validation
if nargin < 1 || isempty(spectra)
    error('apply_sg_smoothing:MissingInput', 'Spectra matrix is required.');
end

if nargin < 2 || isempty(window_size)
    window_size = 11;  % Default for brain tissue FTIR
end

if nargin < 3 || isempty(polynomial_order)
    polynomial_order = 2;  % Default quadratic fit
end

if ~isnumeric(spectra) || ~ismatrix(spectra)
    error('apply_sg_smoothing:InvalidInput', 'Spectra must be a numeric matrix.');
end

%% Validate SG filter parameters
if ~isnumeric(window_size) || window_size < 1 || mod(window_size, 1) ~= 0
    error('apply_sg_smoothing:InvalidWindow', 'Window size must be a positive integer.');
end

if mod(window_size, 2) == 0
    error('apply_sg_smoothing:InvalidWindow', ...
        'Window size must be odd (got %d). Try %d or %d.', ...
        window_size, window_size - 1, window_size + 1);
end

if ~isnumeric(polynomial_order) || polynomial_order < 0 || mod(polynomial_order, 1) ~= 0
    error('apply_sg_smoothing:InvalidPolyOrder', 'Polynomial order must be a non-negative integer.');
end

if polynomial_order >= window_size
    error('apply_sg_smoothing:InvalidParameters', ...
        'Polynomial order (%d) must be less than window size (%d).', ...
        polynomial_order, window_size);
end

%% Check data dimensions
num_spectra = size(spectra, 1);
num_wavenumbers = size(spectra, 2);

if num_wavenumbers < window_size
    warning('apply_sg_smoothing:ShortSpectrum', ...
        'Spectrum length (%d) is less than window size (%d). Returning original spectra.', ...
        num_wavenumbers, window_size);
    smoothed_spectra = spectra;
    return;
end

%% Check for NaN/Inf in input
if any(isnan(spectra(:)) | isinf(spectra(:)))
    warning('apply_sg_smoothing:NaNInfInput', ...
        'NaN or Inf values detected in input spectra. Filter may not work correctly.');
end

%% Initialize output
smoothed_spectra = zeros(num_spectra, num_wavenumbers);

%% Apply Savitzky-Golay smoothing to each spectrum
derivative_order = 0;  % Smoothing only, no derivative

for i = 1:num_spectra
    spectrum = spectra(i, :);
    
    % Check for all-NaN or all-zero spectrum
    if all(isnan(spectrum)) || all(spectrum == 0)
        smoothed_spectra(i, :) = spectrum;
        if i == 1 || mod(i, 100) == 0  % Limit warning spam
            warning('apply_sg_smoothing:InvalidSpectrum', ...
                'Spectrum %d is all-NaN or all-zero. Skipping smoothing.', i);
        end
        continue;
    end
    
    try
        % Apply Savitzky-Golay filter for smoothing (derivative order = 0)
        % sgolayfilt(x, order, framelen) for smoothing
        % Note: sgolayfilt works on columns, so transpose spectrum
        spectrum_col = spectrum';  % Convert to column vector
        smoothed_col = sgolayfilt(spectrum_col, polynomial_order, window_size);
        smoothed_spectra(i, :) = smoothed_col';  % Convert back to row vector
    catch ME
        % If filtering fails, return original spectrum
        smoothed_spectra(i, :) = spectrum;
        warning('apply_sg_smoothing:FilterFailed', ...
            'Smoothing failed for spectrum %d: %s. Returning original spectrum.', ...
            i, ME.message);
    end
end

%% Check output quality
if any(isnan(smoothed_spectra(:)) | isinf(smoothed_spectra(:)))
    warning('apply_sg_smoothing:NaNInfOutput', ...
        'NaN or Inf values detected in smoothed spectra.');
end

end
