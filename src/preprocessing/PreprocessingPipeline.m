classdef PreprocessingPipeline < handle
    %PREPROCESSINGPIPELINE Spectral preprocessing with fit/transform separation
    %
    % DESCRIPTION:
    %   Implements preprocessing transformations with critical separation between
    %   fit_transform (learns parameters from training data) and transform
    %   (applies learned parameters to test data) to prevent data leakage.
    %
    % BSNCX NOTATION:
    %   Preprocessing permutations encoded as 6-character strings:
    %   Position 1 (B): Binning factor (1=none, 2-9=bin size)
    %   Position 2 (S): Smoothing order (0=none, 1-3=Savitzky-Golay order)
    %   Position 3 (N): Normalization (0=none, 2=vector normalization)
    %   Position 4 (C): Correction (0=none, 1=1st derivative, 2=2nd derivative)
    %   Position 5 (X): Reserved for future use
    %
    %   Examples:
    %   '10000X' - No preprocessing (baseline)
    %   '10020X' - Normalization only
    %   '10022X' - Normalization + 2nd derivative
    %   '21022X' - Bin=2, Smooth=1, Norm, 2nd deriv
    %
    % USAGE:
    %   % Training phase
    %   pipeline = PreprocessingPipeline('10022X');
    %   [X_train_proc, params] = pipeline.fit_transform(X_train);
    %   
    %   % Test phase (uses learned parameters)
    %   X_test_proc = pipeline.transform(X_test, params);
    %
    % CRITICAL: Never call fit_transform on test data!
    %
    % See also: Config, DataLoader, CrossValidationEngine
    
    properties (Access = private)
        permutation_id   % BSNCX string
        operations       % Ordered list of operations
        verbose          % Display progress
    end
    
    methods
        function obj = PreprocessingPipeline(permutation_id, varargin)
            %PREPROCESSINGPIPELINE Constructor
            %
            % SYNTAX:
            %   pipeline = PreprocessingPipeline('10022X')
            %   pipeline = PreprocessingPipeline('10022X', 'Verbose', true)
            
            % Parse inputs
            p = inputParser;
            addRequired(p, 'permutation_id', @(x) ischar(x) || isstring(x));
            addParameter(p, 'Verbose', false, @islogical);
            parse(p, permutation_id, varargin{:});
            
            obj.permutation_id = char(p.Results.permutation_id);
            obj.verbose = p.Results.Verbose;
            
            % Validate and parse permutation
            obj.validatePermutation();
            obj.operations = obj.parsePermutation();
            
            if obj.verbose
                fprintf('[PreprocessingPipeline] Created: %s\n', obj.permutation_id);
                fprintf('  Operations: %s\n', strjoin(obj.operations, ' -> '));
            end
        end
        
        function [data_out, params] = fit_transform(obj, data_in)
            %FIT_TRANSFORM Learn parameters from data and transform
            %
            % SYNTAX:
            %   [X_transformed, params] = pipeline.fit_transform(X_train)
            %
            % CRITICAL: Only call on TRAINING data!
            
            validateattributes(data_in, {'double'}, {'2d', 'finite'}, ...
                'fit_transform', 'data_in');
            
            if obj.verbose
                fprintf('[fit_transform] Input: %d x %d\n', ...
                    size(data_in, 1), size(data_in, 2));
            end
            
            data_out = data_in;
            params = struct();
            params.permutation_id = obj.permutation_id;
            params.operations = obj.operations;
            params.dimension_log = {};
            
            % Apply each operation sequentially
            for i = 1:length(obj.operations)
                op = obj.operations{i};
                size_before = size(data_out);
                
                switch op
                    case 'binning'
                        [data_out, params.binning] = obj.apply_binning_fit(data_out);
                        
                    case 'smoothing'
                        [data_out, params.smoothing] = obj.apply_smoothing_fit(data_out);
                        
                    case 'normalization'
                        [data_out, params.normalization] = obj.apply_normalization_fit(data_out);
                        
                    case 'derivative_1'
                        [data_out, params.derivative] = obj.apply_derivative_fit(data_out, 1);
                        
                    case 'derivative_2'
                        [data_out, params.derivative] = obj.apply_derivative_fit(data_out, 2);
                end
                
                % Log dimension changes
                params.dimension_log{end+1} = sprintf('%s: [%d x %d] -> [%d x %d]', ...
                    op, size_before(1), size_before(2), size(data_out, 1), size(data_out, 2));
                
                if obj.verbose
                    fprintf('  %s\n', params.dimension_log{end});
                end
            end
            
            params.final_dimensions = size(data_out);
            
            if obj.verbose
                fprintf('[fit_transform] Output: %d x %d\n', ...
                    size(data_out, 1), size(data_out, 2));
            end
        end
        
        function data_out = transform(obj, data_in, params)
            %TRANSFORM Apply learned parameters to new data
            %
            % SYNTAX:
            %   X_test_transformed = pipeline.transform(X_test, params)
            %
            % CRITICAL: Uses parameters learned from training data only!
            
            validateattributes(data_in, {'double'}, {'2d', 'finite'}, ...
                'transform', 'data_in');
            
            % Validate permutation match
            if ~strcmp(params.permutation_id, obj.permutation_id)
                error('PreprocessingPipeline:PermutationMismatch', ...
                    'Parameter permutation (%s) does not match pipeline (%s)', ...
                    params.permutation_id, obj.permutation_id);
            end
            
            if obj.verbose
                fprintf('[transform] Input: %d x %d\n', ...
                    size(data_in, 1), size(data_in, 2));
            end
            
            data_out = data_in;
            
            % Apply each operation with frozen parameters
            for i = 1:length(params.operations)
                op = params.operations{i};
                
                switch op
                    case 'binning'
                        data_out = obj.apply_binning_transform(data_out, params.binning);
                        
                    case 'smoothing'
                        data_out = obj.apply_smoothing_transform(data_out, params.smoothing);
                        
                    case 'normalization'
                        data_out = obj.apply_normalization_transform(data_out, params.normalization);
                        
                    case 'derivative_1'
                        data_out = obj.apply_derivative_transform(data_out, params.derivative);
                        
                    case 'derivative_2'
                        data_out = obj.apply_derivative_transform(data_out, params.derivative);
                end
            end
            
            % Validate final dimensions match
            if ~isequal(size(data_out, 2), params.final_dimensions(2))
                error('PreprocessingPipeline:DimensionMismatch', ...
                    'Output features (%d) do not match training (%d)', ...
                    size(data_out, 2), params.final_dimensions(2));
            end
            
            if obj.verbose
                fprintf('[transform] Output: %d x %d\n', ...
                    size(data_out, 1), size(data_out, 2));
            end
        end
    end
    
    methods (Access = private)
        function validatePermutation(obj)
            %VALIDATEPERMUTATION Validate permutation string format
            
            perm = obj.permutation_id;
            
            % Check length
            if length(perm) ~= 6
                error('PreprocessingPipeline:InvalidPermutation', ...
                    'Permutation must be 6 characters (BSNCXX format), got: %s', perm);
            end
            
            % Check last character
            if perm(6) ~= 'X'
                error('PreprocessingPipeline:InvalidPermutation', ...
                    'Permutation must end with X, got: %s', perm);
            end
            
            % Check numeric parts
            for i = 1:5
                if ~ismember(perm(i), '0123456789')
                    error('PreprocessingPipeline:InvalidPermutation', ...
                        'Positions 1-5 must be numeric, got: %s', perm);
                end
            end
        end
        
        function ops = parsePermutation(obj)
            %PARSEPERMUTATION Convert BSNCX string to ordered operations
            
            perm = obj.permutation_id;
            ops = {};
            
            B = str2double(perm(1));  % Binning
            S = str2double(perm(2));  % Smoothing
            N = str2double(perm(3));  % Normalization
            C = str2double(perm(4));  % Correction (derivative)
            
            % Build operation sequence
            if B > 1
                ops{end+1} = 'binning';
            end
            
            if S > 0
                ops{end+1} = 'smoothing';
            end
            
            if N == 2
                ops{end+1} = 'normalization';
            end
            
            if C == 1
                ops{end+1} = 'derivative_1';
            elseif C == 2
                ops{end+1} = 'derivative_2';
            end
        end
        
        %% BINNING
        function [data_out, params] = apply_binning_fit(obj, data_in)
            %APPLY_BINNING_FIT Learn binning parameters and transform
            
            perm = obj.permutation_id;
            bin_factor = str2double(perm(1));
            
            n_features = size(data_in, 2);
            n_bins = floor(n_features / bin_factor);
            
            params.bin_factor = bin_factor;
            params.n_bins = n_bins;
            params.n_features_in = n_features;
            
            % Apply binning
            data_out = zeros(size(data_in, 1), n_bins);
            
            for i = 1:n_bins
                start_idx = (i-1) * bin_factor + 1;
                end_idx = min(i * bin_factor, n_features);
                data_out(:, i) = mean(data_in(:, start_idx:end_idx), 2);
            end
        end
        
        function data_out = apply_binning_transform(obj, data_in, params)
            %APPLY_BINNING_TRANSFORM Apply learned binning parameters
            
            n_bins = params.n_bins;
            bin_factor = params.bin_factor;
            n_features = size(data_in, 2);
            
            data_out = zeros(size(data_in, 1), n_bins);
            
            for i = 1:n_bins
                start_idx = (i-1) * bin_factor + 1;
                end_idx = min(i * bin_factor, n_features);
                data_out(:, i) = mean(data_in(:, start_idx:end_idx), 2);
            end
        end
        
        %% SMOOTHING
        function [data_out, params] = apply_smoothing_fit(obj, data_in)
            %APPLY_SMOOTHING_FIT Learn smoothing parameters and transform
            
            perm = obj.permutation_id;
            order = str2double(perm(2));
            
            % Savitzky-Golay parameters
            window_size = max(5, 2*order + 1);
            if mod(window_size, 2) == 0
                window_size = window_size + 1;  % Must be odd
            end
            poly_order = min(order, window_size - 1);
            
            params.order = order;
            params.window_size = window_size;
            params.poly_order = poly_order;
            
            % Apply smoothing
            data_out = sgolayfilt(data_in', poly_order, window_size)';
        end
        
        function data_out = apply_smoothing_transform(obj, data_in, params)
            %APPLY_SMOOTHING_TRANSFORM Apply learned smoothing parameters
            
            data_out = sgolayfilt(data_in', params.poly_order, params.window_size)';
        end
        
        %% NORMALIZATION
        function [data_out, params] = apply_normalization_fit(obj, data_in)
            %APPLY_NORMALIZATION_FIT Learn normalization parameters and transform
            %
            % CRITICAL: Computes mean and std from TRAINING data only!
            % Uses Z-score normalization (standardization)
            
            % Compute mean and std across all training data (global)
            params.mean = mean(data_in, 1);  % Mean per feature
            params.std = std(data_in, 0, 1);  % Std per feature
            
            % Handle zero std (constant features)
            zero_std_idx = params.std == 0;
            if any(zero_std_idx)
                warning('PreprocessingPipeline:ZeroStd', ...
                    '%d features have zero std, setting to 1', sum(zero_std_idx));
                params.std(zero_std_idx) = 1;
            end
            
            % Apply normalization (z-score)
            data_out = (data_in - params.mean) ./ params.std;
        end
        
        function data_out = apply_normalization_transform(obj, data_in, params)
            %APPLY_NORMALIZATION_TRANSFORM Apply learned normalization parameters
            %
            % CRITICAL: Uses mean/std from TRAINING data!
            
            data_out = (data_in - params.mean) ./ params.std;
        end
        
        %% DERIVATIVE
        function [data_out, params] = apply_derivative_fit(obj, data_in, order)
            %APPLY_DERIVATIVE_FIT Compute derivative (no parameters to learn)
            
            params.order = order;
            params.n_features_in = size(data_in, 2);
            params.n_features_out = size(data_in, 2) - order;
            
            % Apply derivative using central difference
            data_out = data_in;
            for i = 1:order
                data_out = diff(data_out, 1, 2);
            end
        end
        
        function data_out = apply_derivative_transform(obj, data_in, params)
            %APPLY_DERIVATIVE_TRANSFORM Apply derivative transformation
            
            data_out = data_in;
            for i = 1:params.order
                data_out = diff(data_out, 1, 2);
            end
        end
    end
end
