function config = create_preprocessing_config(approach)
% CREATE_PREPROCESSING_CONFIG Generate preprocessing configuration structure
%
% SYNTAX:
%   config = create_preprocessing_config(approach)
%
% INPUTS:
%   approach - String specifying preprocessing approach:
%              'PP1' or 'approach1' - Standard pipeline (default for analysis)
%              'PP2' or 'approach2' - Enhanced pipeline with binning/smoothing
%              'custom'             - Returns empty config for manual setup
%
% OUTPUTS:
%   config - Structure with preprocessing parameters:
%            .name                - Approach name ('PP1' or 'PP2')
%            .description         - Brief description
%            .apply_binning       - Logical flag
%            .bin_factor          - Binning factor (if apply_binning = true)
%            .apply_smoothing     - Logical flag
%            .smooth_window       - SG smoothing window size
%            .smooth_poly_order   - SG smoothing polynomial order
%            .apply_normalization - Logical flag
%            .norm_type           - Normalization type ('L2')
%            .apply_derivative    - Logical flag
%            .deriv_window        - SG derivative window size
%            .deriv_poly_order    - SG derivative polynomial order
%            .deriv_order         - Derivative order (0=none, 1=first, 2=second)
%
% DESCRIPTION:
%   Creates standardized preprocessing configuration structures for the
%   two main preprocessing approaches used in the meningioma FTIR pipeline.
%   
%   APPROACH 1 (PP1) - Standard Pipeline (DEFAULT):
%   - No binning (full spectral resolution)
%   - No smoothing
%   - Vector normalization (L2 norm = 1)
%   - Second derivative baseline correction (SG: window=5, poly=2, deriv=2)
%   
%   APPROACH 2 (PP2) - Enhanced Pipeline:
%   - Binning (factor 4)
%   - SG smoothing (window=11, poly=2, deriv=0)
%   - Vector normalization (L2 norm = 1)
%   - Second derivative baseline correction (SG: window=5, poly=2, deriv=2)
%
% EXAMPLE:
%   % Get PP1 configuration
%   cfg = create_preprocessing_config('PP1');
%   
%   % Get PP2 configuration
%   cfg = create_preprocessing_config('PP2');
%   
%   % Create custom configuration
%   cfg = create_preprocessing_config('custom');
%   cfg.apply_binning = true;
%   cfg.bin_factor = 2;  % Custom binning factor
%
% See also: PREPROCESS_SPECTRA, APPLY_BINNING, APPLY_VECTOR_NORMALIZATION
%
% Author: GitHub Copilot
% Date: 2025-10-24

%% Input validation
if nargin < 1 || isempty(approach)
    approach = 'PP1';  % Default to standard pipeline
end

% Normalize input string
approach = char(approach);
approach_lower = lower(strtrim(approach));

%% Initialize config structure
config = struct();
config.name = '';
config.description = '';
config.apply_binning = false;
config.bin_factor = 1;
config.apply_smoothing = false;
config.smooth_window = 11;
config.smooth_poly_order = 2;
config.apply_normalization = true;  % Always applied in both approaches
config.norm_type = 'L2';  % L2 normalization (unit vector)
config.apply_derivative = true;  % Always applied in both approaches
config.deriv_window = 5;
config.deriv_poly_order = 2;
config.deriv_order = 2;  % Second derivative

%% Set approach-specific parameters
switch approach_lower
    case {'pp1', 'approach1', '1'}
        % APPROACH 1 (PP1) - Standard Pipeline
        config.name = 'PP1';
        config.description = 'Standard pipeline: L2 norm + 2nd derivative';
        
        % No binning
        config.apply_binning = false;
        config.bin_factor = 1;
        
        % No smoothing
        config.apply_smoothing = false;
        
        % Vector normalization (L2)
        config.apply_normalization = true;
        config.norm_type = 'L2';
        
        % Second derivative baseline correction
        config.apply_derivative = true;
        config.deriv_window = 5;
        config.deriv_poly_order = 2;
        config.deriv_order = 2;
        
    case {'pp2', 'approach2', '2'}
        % APPROACH 2 (PP2) - Enhanced Pipeline
        config.name = 'PP2';
        config.description = 'Enhanced pipeline: Bin + Smooth + L2 norm + 2nd derivative';
        
        % Binning (factor 4)
        config.apply_binning = true;
        config.bin_factor = 4;
        
        % SG smoothing (NOT derivative)
        config.apply_smoothing = true;
        config.smooth_window = 11;
        config.smooth_poly_order = 2;
        
        % Vector normalization (L2)
        config.apply_normalization = true;
        config.norm_type = 'L2';
        
        % Second derivative baseline correction
        config.apply_derivative = true;
        config.deriv_window = 5;
        config.deriv_poly_order = 2;
        config.deriv_order = 2;
        
    case {'custom', 'empty'}
        % Custom configuration - user will set parameters manually
        config.name = 'Custom';
        config.description = 'User-defined preprocessing configuration';
        % All parameters already initialized to defaults above
        
    otherwise
        error('create_preprocessing_config:InvalidApproach', ...
            ['Unknown approach "%s". Valid options: ''PP1'', ''PP2'', or ''custom''. ', ...
             'Use ''PP1'' for standard pipeline (default) or ''PP2'' for enhanced pipeline.'], ...
            approach);
end

%% Validate configuration
validate_config(config);

end

%% Helper function to validate configuration
function validate_config(cfg)
    % Check required fields
    required_fields = {'name', 'description', 'apply_binning', 'bin_factor', ...
                      'apply_smoothing', 'smooth_window', 'smooth_poly_order', ...
                      'apply_normalization', 'norm_type', 'apply_derivative', ...
                      'deriv_window', 'deriv_poly_order', 'deriv_order'};
    
    missing_fields = setdiff(required_fields, fieldnames(cfg));
    if ~isempty(missing_fields)
        error('create_preprocessing_config:MissingFields', ...
            'Configuration is missing required fields: %s', ...
            strjoin(missing_fields, ', '));
    end
    
    % Validate parameter values
    if cfg.apply_binning
        if cfg.bin_factor < 1 || mod(cfg.bin_factor, 1) ~= 0
            error('create_preprocessing_config:InvalidBinFactor', ...
                'Bin factor must be a positive integer (got %.2f).', cfg.bin_factor);
        end
    end
    
    if cfg.apply_smoothing
        if mod(cfg.smooth_window, 2) == 0
            error('create_preprocessing_config:InvalidSmoothWindow', ...
                'Smoothing window must be odd (got %d).', cfg.smooth_window);
        end
        if cfg.smooth_poly_order >= cfg.smooth_window
            error('create_preprocessing_config:InvalidSmoothParams', ...
                'Smoothing polynomial order (%d) must be less than window size (%d).', ...
                cfg.smooth_poly_order, cfg.smooth_window);
        end
    end
    
    if cfg.apply_derivative
        if mod(cfg.deriv_window, 2) == 0
            error('create_preprocessing_config:InvalidDerivWindow', ...
                'Derivative window must be odd (got %d).', cfg.deriv_window);
        end
        if cfg.deriv_poly_order >= cfg.deriv_window
            error('create_preprocessing_config:InvalidDerivParams', ...
                'Derivative polynomial order (%d) must be less than window size (%d).', ...
                cfg.deriv_poly_order, cfg.deriv_window);
        end
        if cfg.deriv_order > cfg.deriv_poly_order
            error('create_preprocessing_config:InvalidDerivOrder', ...
                'Derivative order (%d) must not exceed polynomial order (%d).', ...
                cfg.deriv_order, cfg.deriv_poly_order);
        end
    end
    
    if cfg.apply_normalization
        valid_norm_types = {'L2', 'SNV', 'MinMax', 'None'};
        if ~ismember(cfg.norm_type, valid_norm_types)
            error('create_preprocessing_config:InvalidNormType', ...
                'Invalid normalization type "%s". Valid options: %s', ...
                cfg.norm_type, strjoin(valid_norm_types, ', '));
        end
    end
end
