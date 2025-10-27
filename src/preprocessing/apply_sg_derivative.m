function derivative_spectra = apply_sg_derivative(spectra, window_size, polynomial_order, derivative_order)
% APPLY_SG_DERIVATIVE Apply Savitzky-Golay derivative filter for baseline correction
%
% SYNTAX:
%   derivative_spectra = apply_sg_derivative(spectra, window_size, polynomial_order, derivative_order)
%
% INPUTS:
%   spectra           - Matrix of spectra (rows = spectra, cols = wavenumbers)
%                       Can also be a single row vector
%   window_size       - Filter window size (must be odd, default: 5)
%   polynomial_order  - Polynomial order for fitting (default: 2)
%   derivative_order  - Order of derivative (default: 2 for second derivative)
%
% OUTPUTS:
%   derivative_spectra - Derivative spectra (same dimensions as input)
%
% DESCRIPTION:
%   Applies Savitzky-Golay derivative filter for baseline correction.
%   Second derivative is commonly used to remove baseline variations and
%   enhance peak resolution in spectroscopy.
%   
%   The filter fits a polynomial to a sliding window and computes the
%   analytical derivative of the fitted polynomial at the center point.
%   
%   Recommended parameters for brain tissue FTIR:
%   - Window size: 5 or 7 points
%   - Polynomial order: 2 or 3 (start with 2)
%   - Derivative order: 2 (second derivative for baseline correction)
%   
%   Constraints:
%   - window_size must be odd
%   - window_size > polynomial_order
%   - window_size <= number of data points
%   - derivative_order <= polynomial_order
%
% EDGE CASES:
%   - Spectra shorter than window: Uses original spectrum with warning
%   - NaN/Inf values: Attempts to filter, may propagate issues
%
% EXAMPLE:
%   % Second derivative with default parameters
%   second_deriv = apply_sg_derivative(my_spectra);
%   
%   % Custom parameters
%   second_deriv = apply_sg_derivative(my_spectra, 7, 3, 2);
%
% See also: APPLY_SG_SMOOTHING, SGOLAYFILT, PREPROCESS_SPECTRA
%
% Author: GitHub Copilot
% Date: 2025-10-24

%% Input validation
if nargin < 1 || isempty(spectra)
    error('apply_sg_derivative:MissingInput', 'Spectra matrix is required.');
end

if nargin < 2 || isempty(window_size)
    window_size = 5;  % Default for brain tissue FTIR
end

if nargin < 3 || isempty(polynomial_order)
    polynomial_order = 2;  % Default quadratic fit
end

if nargin < 4 || isempty(derivative_order)
    derivative_order = 2;  % Default second derivative
end

if ~isnumeric(spectra) || ~ismatrix(spectra)
    error('apply_sg_derivative:InvalidInput', 'Spectra must be a numeric matrix.');
end

%% Validate SG filter parameters
if ~isnumeric(window_size) || window_size < 1 || mod(window_size, 1) ~= 0
    error('apply_sg_derivative:InvalidWindow', 'Window size must be a positive integer.');
end

if mod(window_size, 2) == 0
    error('apply_sg_derivative:InvalidWindow', ...
        'Window size must be odd (got %d). Try %d or %d.', ...
        window_size, window_size - 1, window_size + 1);
end

if ~isnumeric(polynomial_order) || polynomial_order < 0 || mod(polynomial_order, 1) ~= 0
    error('apply_sg_derivative:InvalidPolyOrder', 'Polynomial order must be a non-negative integer.');
end

if polynomial_order >= window_size
    error('apply_sg_derivative:InvalidParameters', ...
        'Polynomial order (%d) must be less than window size (%d).', ...
        polynomial_order, window_size);
end

if ~isnumeric(derivative_order) || derivative_order < 0 || mod(derivative_order, 1) ~= 0
    error('apply_sg_derivative:InvalidDerivOrder', 'Derivative order must be a non-negative integer.');
end

if derivative_order > polynomial_order
    error('apply_sg_derivative:InvalidParameters', ...
        'Derivative order (%d) must not exceed polynomial order (%d).', ...
        derivative_order, polynomial_order);
end

%% Check data dimensions
num_spectra = size(spectra, 1);
num_wavenumbers = size(spectra, 2);

if num_wavenumbers < window_size
    warning('apply_sg_derivative:ShortSpectrum', ...
        'Spectrum length (%d) is less than window size (%d). Returning original spectra.', ...
        num_wavenumbers, window_size);
    derivative_spectra = spectra;
    return;
end

%% Check for NaN/Inf in input
if any(isnan(spectra(:)) | isinf(spectra(:)))
    warning('apply_sg_derivative:NaNInfInput', ...
        'NaN or Inf values detected in input spectra. Filter may not work correctly.');
end

%% Initialize output
derivative_spectra = zeros(num_spectra, num_wavenumbers);

%% Apply Savitzky-Golay derivative to each spectrum
for i = 1:num_spectra
    spectrum = spectra(i, :);
    
    % Check for all-NaN or all-zero spectrum
    if all(isnan(spectrum)) || all(spectrum == 0)
        derivative_spectra(i, :) = spectrum;
        if i == 1 || mod(i, 100) == 0  % Limit warning spam
            warning('apply_sg_derivative:InvalidSpectrum', ...
                'Spectrum %d is all-NaN or all-zero. Skipping derivative calculation.', i);
        end
        continue;
    end
    
    try
        % Apply Savitzky-Golay derivative filter
        % For derivatives, we need to use sgolay to get the filter coefficients
        % then apply them manually
        spectrum_col = spectrum';  % Convert to column vector
        
        % Get SG filter coefficients
        [b, g] = sgolay(polynomial_order, window_size);
        
        % Apply the derivative filter (g contains derivative coefficients)
        % The derivative_order+1 column contains the coefficients for that derivative
        dt = 1;  % Spacing between points (assume unit spacing)
        deriv_col = conv(spectrum_col, factorial(derivative_order) * g(:, derivative_order+1) / (dt^derivative_order), 'same');
        
        derivative_spectra(i, :) = deriv_col';  % Convert back to row vector
    catch ME
        % If filtering fails, return original spectrum
        derivative_spectra(i, :) = spectrum;
        warning('apply_sg_derivative:FilterFailed', ...
            'Derivative calculation failed for spectrum %d: %s. Returning original spectrum.', ...
            i, ME.message);
    end
end

%% Check output quality
if any(isnan(derivative_spectra(:)) | isinf(derivative_spectra(:)))
    warning('apply_sg_derivative:NaNInfOutput', ...
        'NaN or Inf values detected in derivative spectra.');
end

%% Informational message for first call
persistent first_call;
if isempty(first_call)
    first_call = false;
    if derivative_order == 2
        fprintf('apply_sg_derivative: Computing 2nd derivative for baseline correction (window=%d, poly=%d).\n', ...
            window_size, polynomial_order);
    end
end

end
