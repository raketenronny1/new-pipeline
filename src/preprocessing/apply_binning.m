function [binned_spectra, binned_wavenumbers] = apply_binning(spectra, wavenumbers, bin_factor)
% APPLY_BINNING Reduce spectral resolution by averaging neighboring points
%
% SYNTAX:
%   [binned_spectra, binned_wavenumbers] = apply_binning(spectra, wavenumbers, bin_factor)
%
% INPUTS:
%   spectra        - Matrix of spectra (rows = spectra, cols = wavenumbers)
%                    Can also be a single row vector
%   wavenumbers    - Row vector of wavenumber values corresponding to columns
%   bin_factor     - Integer binning factor (default: 4)
%                    Specifies how many consecutive points to average
%
% OUTPUTS:
%   binned_spectra     - Binned spectra matrix (reduced column dimension)
%   binned_wavenumbers - Binned wavenumber vector (averaged within each bin)
%
% DESCRIPTION:
%   Reduces spectral resolution by averaging consecutive wavenumber points
%   in groups of bin_factor. This improves SNR and reduces computational load
%   at the expense of spectral resolution.
%
%   Example: bin_factor = 4 reduces 441 points to 110 points
%
% EXAMPLE:
%   % Bin spectra with factor 4
%   [binned, wn_binned] = apply_binning(my_spectra, wavenumbers, 4);
%
% See also: APPLY_SG_SMOOTHING, PREPROCESS_SPECTRA
%
% Author: GitHub Copilot
% Date: 2025-10-24

%% Input validation
if nargin < 3 || isempty(bin_factor)
    bin_factor = 4;  % Default binning factor
end

if nargin < 2 || isempty(wavenumbers)
    error('apply_binning:MissingInput', 'Wavenumber vector is required.');
end

if ~isnumeric(spectra) || ~ismatrix(spectra)
    error('apply_binning:InvalidInput', 'Spectra must be a numeric matrix.');
end

if ~isvector(wavenumbers)
    error('apply_binning:InvalidInput', 'Wavenumbers must be a vector.');
end

if ~isnumeric(bin_factor) || bin_factor < 1 || mod(bin_factor, 1) ~= 0
    error('apply_binning:InvalidBinFactor', 'Bin factor must be a positive integer.');
end

% Ensure wavenumbers is a row vector
if iscolumn(wavenumbers)
    wavenumbers = wavenumbers';
end

% Ensure spectra has correct number of columns
num_wavenumbers = length(wavenumbers);
if size(spectra, 2) ~= num_wavenumbers
    error('apply_binning:DimensionMismatch', ...
        'Number of columns in spectra (%d) must match wavenumber vector length (%d).', ...
        size(spectra, 2), num_wavenumbers);
end

%% Handle trivial case
if bin_factor == 1
    binned_spectra = spectra;
    binned_wavenumbers = wavenumbers;
    return;
end

%% Calculate number of bins
num_bins = floor(num_wavenumbers / bin_factor);

if num_bins == 0
    error('apply_binning:TooFewPoints', ...
        'Bin factor (%d) is too large for spectrum length (%d).', ...
        bin_factor, num_wavenumbers);
end

% Number of points to use (may exclude trailing points if not divisible)
num_points_to_use = num_bins * bin_factor;

%% Initialize output arrays
num_spectra = size(spectra, 1);
binned_spectra = zeros(num_spectra, num_bins);
binned_wavenumbers = zeros(1, num_bins);

%% Perform binning
for i = 1:num_bins
    % Calculate indices for this bin
    bin_start = (i - 1) * bin_factor + 1;
    bin_end = i * bin_factor;
    
    % Average spectra within bin (across columns, for all rows)
    binned_spectra(:, i) = mean(spectra(:, bin_start:bin_end), 2, 'omitnan');
    
    % Average wavenumbers within bin
    binned_wavenumbers(i) = mean(wavenumbers(bin_start:bin_end), 'omitnan');
end

%% Warning if points were excluded
if num_points_to_use < num_wavenumbers
    points_excluded = num_wavenumbers - num_points_to_use;
    warning('apply_binning:PointsExcluded', ...
        '%d trailing wavenumber points excluded (not divisible by bin factor %d).', ...
        points_excluded, bin_factor);
end

%% Handle NaN/Inf in output
if any(isnan(binned_spectra(:)) | isinf(binned_spectra(:)))
    warning('apply_binning:NaNInfDetected', ...
        'NaN or Inf values detected in binned spectra (may be present in input).');
end

end
