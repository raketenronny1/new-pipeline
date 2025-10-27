function normalized_spectra = apply_vector_normalization(spectra)
% APPLY_VECTOR_NORMALIZATION Normalize spectra to unit L2 norm
%
% SYNTAX:
%   normalized_spectra = apply_vector_normalization(spectra)
%
% INPUTS:
%   spectra - Matrix of spectra (rows = spectra, cols = wavenumbers)
%             Can also be a single row vector
%
% OUTPUTS:
%   normalized_spectra - Normalized spectra where each row has L2 norm = 1
%
% DESCRIPTION:
%   Performs vector normalization (L2 normalization) on each spectrum.
%   Each spectrum vector is divided by its Euclidean norm (L2 norm).
%   
%   Formula: normalized_spectrum = spectrum / norm(spectrum, 2)
%   
%   This removes intensity scaling effects and focuses analysis on spectral
%   shape rather than absolute intensity. Also known as "unit vector 
%   normalization" or "Euclidean normalization".
%   
%   NOTE: This is NOT Standard Normal Variate (SNV) normalization.
%   SNV uses: (spectrum - mean) / std
%   L2 uses:  spectrum / sqrt(sum(spectrum.^2))
%
% EDGE CASES:
%   - Zero vectors: Returns zero vector with warning
%   - NaN/Inf values: Propagated to output with warning
%   - Near-zero norm: Returns original spectrum with warning
%
% EXAMPLE:
%   % Normalize spectra to unit length
%   normalized = apply_vector_normalization(my_spectra);
%   
%   % Verify: L2 norm should be 1.0
%   norms = sqrt(sum(normalized.^2, 2));  % Should be all 1.0
%
% See also: APPLY_BINNING, APPLY_SG_DERIVATIVE, PREPROCESS_SPECTRA
%
% Author: GitHub Copilot
% Date: 2025-10-24

%% Input validation
if nargin < 1 || isempty(spectra)
    error('apply_vector_normalization:MissingInput', 'Spectra matrix is required.');
end

if ~isnumeric(spectra) || ~ismatrix(spectra)
    error('apply_vector_normalization:InvalidInput', 'Spectra must be a numeric matrix.');
end

%% Check for NaN/Inf in input
if any(isnan(spectra(:)) | isinf(spectra(:)))
    warning('apply_vector_normalization:NaNInfInput', ...
        'NaN or Inf values detected in input spectra. These will propagate to output.');
end

%% Initialize output
num_spectra = size(spectra, 1);
num_wavenumbers = size(spectra, 2);
normalized_spectra = zeros(num_spectra, num_wavenumbers);

%% Normalize each spectrum
for i = 1:num_spectra
    spectrum = spectra(i, :);
    
    % Calculate L2 norm (Euclidean norm)
    spectrum_norm = norm(spectrum, 2);
    
    % Handle edge cases
    if isnan(spectrum_norm) || isinf(spectrum_norm)
        % NaN or Inf norm - return original with warning
        normalized_spectra(i, :) = spectrum;
        warning('apply_vector_normalization:InvalidNorm', ...
            'Spectrum %d has NaN or Inf norm. Returning original spectrum.', i);
        
    elseif spectrum_norm < eps(1)  % Essentially zero (machine precision)
        % Zero or near-zero vector - cannot normalize
        normalized_spectra(i, :) = spectrum;
        if i == 1 || mod(i, 100) == 0  % Limit warning spam
            warning('apply_vector_normalization:ZeroNorm', ...
                'Spectrum %d has zero or near-zero norm (%.2e). Returning original spectrum.', ...
                i, spectrum_norm);
        end
        
    else
        % Normal case: divide by L2 norm
        normalized_spectra(i, :) = spectrum / spectrum_norm;
    end
end

%% Verification (optional, can be commented out for performance)
% Verify that norms are approximately 1.0 for successfully normalized spectra
if nargout == 0 || nargout == 1
    % Only verify if user might see warnings (not when called internally in loop)
    sample_indices = unique([1, round(num_spectra/2), num_spectra]);  % Check first, middle, last
    sample_indices = sample_indices(sample_indices <= num_spectra);
    
    for i = sample_indices
        test_norm = norm(normalized_spectra(i, :), 2);
        if ~isnan(test_norm) && ~isinf(test_norm) && abs(test_norm) > eps(1)
            % Should be very close to 1.0
            if abs(test_norm - 1.0) > 1e-6
                warning('apply_vector_normalization:NormCheckFailed', ...
                    'Spectrum %d: normalized L2 norm = %.6f (expected 1.0).', i, test_norm);
            end
        end
    end
end

end
