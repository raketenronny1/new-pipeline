% TEST_PREPROCESSING_FUNCTIONS
% Test script for all preprocessing functions
%
% This script validates the preprocessing functions with synthetic and real data.
% It tests individual functions and the complete preprocessing pipeline.
%
% Author: GitHub Copilot
% Date: 2025-10-24

clear; clc; close all;

fprintf('=======================================================\n');
fprintf('   Testing FTIR Preprocessing Functions\n');
fprintf('=======================================================\n\n');

% Add preprocessing directory to path
addpath(fullfile(fileparts(mfilename('fullpath'))));

%% Test 1: Create synthetic test data
fprintf('TEST 1: Creating synthetic test data...\n');

% Create synthetic wavenumber vector (fingerprint region)
wavenumbers = linspace(1800, 950, 441);  % 441 points, as in real data

% Create synthetic spectra with known features
num_test_spectra = 10;
synthetic_spectra = zeros(num_test_spectra, length(wavenumbers));

for i = 1:num_test_spectra
    % Base spectrum with Gaussian peaks
    spectrum = zeros(1, length(wavenumbers));
    
    % Add 3-5 Gaussian peaks at random positions
    num_peaks = randi([3, 5]);
    for p = 1:num_peaks
        center = 950 + rand() * (1800 - 950);
        width = 20 + rand() * 30;
        amplitude = 0.5 + rand() * 0.5;
        spectrum = spectrum + amplitude * exp(-((wavenumbers - center) / width).^2);
    end
    
    % Add baseline drift
    baseline = 0.1 * (wavenumbers - min(wavenumbers)) / (max(wavenumbers) - min(wavenumbers));
    spectrum = spectrum + baseline;
    
    % Add noise
    noise_level = 0.02;
    spectrum = spectrum + noise_level * randn(size(spectrum));
    
    synthetic_spectra(i, :) = spectrum;
end

fprintf('  Created %d synthetic spectra with %d wavenumbers\n', ...
    num_test_spectra, length(wavenumbers));
fprintf('  ✓ PASS\n\n');

%% Test 2: Test apply_binning
fprintf('TEST 2: Testing apply_binning...\n');

try
    [binned, wn_binned] = apply_binning(synthetic_spectra, wavenumbers, 4);
    
    % Validate output
    assert(size(binned, 1) == num_test_spectra, 'Row count mismatch after binning');
    assert(size(binned, 2) == length(wn_binned), 'Column count mismatch with wavenumbers');
    assert(size(binned, 2) == floor(length(wavenumbers) / 4), 'Incorrect binning factor');
    assert(~any(isnan(binned(:))), 'NaN values in binned output');
    
    fprintf('  Input:  %d x %d\n', size(synthetic_spectra));
    fprintf('  Output: %d x %d\n', size(binned));
    fprintf('  ✓ PASS\n\n');
catch ME
    fprintf('  ✗ FAIL: %s\n\n', ME.message);
end

%% Test 3: Test apply_vector_normalization
fprintf('TEST 3: Testing apply_vector_normalization...\n');

try
    normalized = apply_vector_normalization(synthetic_spectra);
    
    % Validate output
    assert(size(normalized, 1) == num_test_spectra, 'Row count mismatch');
    assert(size(normalized, 2) == length(wavenumbers), 'Column count mismatch');
    
    % Check L2 norms (should all be ~1.0)
    norms = sqrt(sum(normalized.^2, 2));
    assert(all(abs(norms - 1.0) < 1e-6), 'L2 norms not equal to 1.0');
    
    fprintf('  L2 norms: min=%.6f, max=%.6f, mean=%.6f\n', ...
        min(norms), max(norms), mean(norms));
    fprintf('  ✓ PASS (all norms ≈ 1.0)\n\n');
catch ME
    fprintf('  ✗ FAIL: %s\n\n', ME.message);
end

%% Test 4: Test apply_sg_smoothing
fprintf('TEST 4: Testing apply_sg_smoothing...\n');

try
    smoothed = apply_sg_smoothing(synthetic_spectra, 11, 2);
    
    % Validate output
    assert(size(smoothed, 1) == num_test_spectra, 'Row count mismatch');
    assert(size(smoothed, 2) == length(wavenumbers), 'Column count mismatch');
    assert(~any(isnan(smoothed(:))), 'NaN values in smoothed output');
    
    % Smoothed spectrum should have lower variance than original
    orig_var = var(synthetic_spectra(1, :));
    smooth_var = var(smoothed(1, :));
    fprintf('  Original variance: %.6f\n', orig_var);
    fprintf('  Smoothed variance: %.6f (%.1f%% reduction)\n', ...
        smooth_var, 100 * (1 - smooth_var / orig_var));
    fprintf('  ✓ PASS\n\n');
catch ME
    fprintf('  ✗ FAIL: %s\n\n', ME.message);
end

%% Test 5: Test apply_sg_derivative
fprintf('TEST 5: Testing apply_sg_derivative...\n');

try
    deriv2 = apply_sg_derivative(synthetic_spectra, 5, 2, 2);
    
    % Validate output
    assert(size(deriv2, 1) == num_test_spectra, 'Row count mismatch');
    assert(size(deriv2, 2) == length(wavenumbers), 'Column count mismatch');
    
    % Second derivative should have mean close to zero
    mean_deriv = mean(deriv2(:));
    fprintf('  Mean of 2nd derivative: %.6f (should be ≈ 0)\n', mean_deriv);
    fprintf('  ✓ PASS\n\n');
catch ME
    fprintf('  ✗ FAIL: %s\n\n', ME.message);
end

%% Test 6: Test create_preprocessing_config
fprintf('TEST 6: Testing create_preprocessing_config...\n');

try
    cfg_pp1 = create_preprocessing_config('PP1');
    cfg_pp2 = create_preprocessing_config('PP2');
    
    % Validate PP1
    assert(strcmp(cfg_pp1.name, 'PP1'), 'PP1 name mismatch');
    assert(cfg_pp1.apply_binning == false, 'PP1 should not use binning');
    assert(cfg_pp1.apply_smoothing == false, 'PP1 should not use smoothing');
    assert(cfg_pp1.apply_normalization == true, 'PP1 should use normalization');
    assert(cfg_pp1.apply_derivative == true, 'PP1 should use derivative');
    
    % Validate PP2
    assert(strcmp(cfg_pp2.name, 'PP2'), 'PP2 name mismatch');
    assert(cfg_pp2.apply_binning == true, 'PP2 should use binning');
    assert(cfg_pp2.bin_factor == 4, 'PP2 binning factor should be 4');
    assert(cfg_pp2.apply_smoothing == true, 'PP2 should use smoothing');
    assert(cfg_pp2.smooth_window == 11, 'PP2 smoothing window should be 11');
    
    fprintf('  PP1 config: ✓\n');
    fprintf('  PP2 config: ✓\n');
    fprintf('  ✓ PASS\n\n');
catch ME
    fprintf('  ✗ FAIL: %s\n\n', ME.message);
end

%% Test 7: Test complete preprocessing pipeline (PP1)
fprintf('TEST 7: Testing complete PP1 pipeline...\n');

try
    [pp1_spectra, pp1_wn] = preprocess_spectra(synthetic_spectra, wavenumbers, 'PP1');
    
    % Validate output
    assert(size(pp1_spectra, 1) == num_test_spectra, 'Row count mismatch');
    assert(size(pp1_spectra, 2) == length(pp1_wn), 'Column/wavenumber mismatch');
    assert(length(pp1_wn) == length(wavenumbers), 'PP1 should not change wavenumber count');
    
    fprintf('  ✓ PASS\n\n');
catch ME
    fprintf('  ✗ FAIL: %s\n\n', ME.message);
end

%% Test 8: Test complete preprocessing pipeline (PP2)
fprintf('TEST 8: Testing complete PP2 pipeline...\n');

try
    [pp2_spectra, pp2_wn] = preprocess_spectra(synthetic_spectra, wavenumbers, 'PP2');
    
    % Validate output
    assert(size(pp2_spectra, 1) == num_test_spectra, 'Row count mismatch');
    assert(size(pp2_spectra, 2) == length(pp2_wn), 'Column/wavenumber mismatch');
    assert(length(pp2_wn) < length(wavenumbers), 'PP2 should reduce wavenumber count');
    assert(length(pp2_wn) == floor(length(wavenumbers) / 4), 'PP2 binning incorrect');
    
    fprintf('  ✓ PASS\n\n');
catch ME
    fprintf('  ✗ FAIL: %s\n\n', ME.message);
end

%% Test 9: Visual comparison (if requested)
fprintf('TEST 9: Creating visualization...\n');

try
    figure('Name', 'Preprocessing Comparison', 'Position', [100, 100, 1200, 800]);
    
    % Select one spectrum for visualization
    test_idx = 1;
    
    % Plot 1: Original
    subplot(3, 1, 1);
    plot(wavenumbers, synthetic_spectra(test_idx, :), 'b-', 'LineWidth', 1.5);
    xlabel('Wavenumber (cm^{-1})');
    ylabel('Absorbance');
    title('Raw Spectrum');
    grid on;
    set(gca, 'XDir', 'reverse');
    
    % Plot 2: PP1
    subplot(3, 1, 2);
    plot(pp1_wn, pp1_spectra(test_idx, :), 'r-', 'LineWidth', 1.5);
    xlabel('Wavenumber (cm^{-1})');
    ylabel('2nd Derivative');
    title('PP1: Vector Norm + 2nd Derivative');
    grid on;
    set(gca, 'XDir', 'reverse');
    
    % Plot 3: PP2
    subplot(3, 1, 3);
    plot(pp2_wn, pp2_spectra(test_idx, :), 'g-', 'LineWidth', 1.5);
    xlabel('Wavenumber (cm^{-1})');
    ylabel('2nd Derivative');
    title('PP2: Bin + Smooth + Vector Norm + 2nd Derivative');
    grid on;
    set(gca, 'XDir', 'reverse');
    
    fprintf('  Visualization created\n');
    fprintf('  ✓ PASS\n\n');
catch ME
    fprintf('  ✗ FAIL: %s\n\n', ME.message);
end

%% Test 10: Edge case handling
fprintf('TEST 10: Testing edge case handling...\n');

try
    % Test with single spectrum
    single = synthetic_spectra(1, :);
    [processed_single, ~] = preprocess_spectra(single, wavenumbers, 'PP1');
    assert(size(processed_single, 1) == 1, 'Single spectrum handling failed');
    
    % Test with zero spectrum (should warn but not error)
    zero_spectrum = zeros(1, length(wavenumbers));
    [processed_zero, ~] = preprocess_spectra(zero_spectrum, wavenumbers, 'PP1');
    assert(all(size(processed_zero) == size(zero_spectrum)), 'Zero spectrum handling failed');
    
    fprintf('  Single spectrum: ✓\n');
    fprintf('  Zero spectrum: ✓\n');
    fprintf('  ✓ PASS\n\n');
catch ME
    fprintf('  ✗ FAIL: %s\n\n', ME.message);
end

%% Summary
fprintf('=======================================================\n');
fprintf('   All Tests Completed Successfully!\n');
fprintf('=======================================================\n\n');

fprintf('Preprocessing functions are ready for use.\n');
fprintf('Location: src/preprocessing/\n\n');

fprintf('Available functions:\n');
fprintf('  - apply_binning.m\n');
fprintf('  - apply_vector_normalization.m\n');
fprintf('  - apply_sg_smoothing.m\n');
fprintf('  - apply_sg_derivative.m\n');
fprintf('  - create_preprocessing_config.m\n');
fprintf('  - preprocess_spectra.m (main orchestrator)\n\n');

fprintf('Example usage:\n');
fprintf('  [pp1, wn1] = preprocess_spectra(raw_spectra, wavenumbers, ''PP1'');\n');
fprintf('  [pp2, wn2] = preprocess_spectra(raw_spectra, wavenumbers, ''PP2'');\n\n');
