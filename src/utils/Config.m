classdef Config < handle
    %CONFIG Singleton configuration manager for preprocessing optimization pipeline
    %
    % DESCRIPTION:
    %   Provides centralized, validated configuration management with singleton
    %   pattern to ensure consistent parameters across the entire pipeline.
    %
    % USAGE:
    %   % Get configuration instance
    %   cfg = Config.getInstance();
    %   
    %   % Load from file
    %   cfg = Config.getInstance('config.json');
    %   
    %   % Load from struct
    %   params = struct('n_folds', 10, 'n_repeats', 5);
    %   cfg = Config.getInstance(params);
    %
    % CRITICAL PARAMETERS:
    %   n_repeats                  - Number of CV repetitions (default: 10)
    %   n_folds                    - Number of CV folds (default: 10)
    %   random_seed                - Random seed for reproducibility (default: 42)
    %   parallel                   - Enable parallel processing (default: true)
    %   classifiers                - Cell array of classifier names
    %   preprocessing_permutations - Cell array of BSNCX strings
    %   pca_variance_threshold     - PCA variance threshold (default: 0.95)
    %
    % See also: DataLoader, PreprocessingPipeline, CrossValidationEngine
    
    properties (Access = private)
        config_struct  % Internal configuration structure
    end
    
    methods (Access = private)
        function obj = Config(source)
            %CONFIG Private constructor (singleton pattern)
            if nargin == 0
                % Use defaults
                obj.config_struct = Config.getDefaultConfig();
            elseif ischar(source) || isstring(source)
                % Load from file
                obj.loadFromFile(source);
            elseif isstruct(source)
                % Load from struct
                obj.loadFromStruct(source);
            else
                error('Config:InvalidInput', ...
                    'Input must be a filename (string) or struct');
            end
            
            % Validate configuration
            obj.validateConfig();
        end
    end
    
    methods (Static)
        function obj = getInstance(source)
            %GETINSTANCE Get or create singleton instance
            %
            % SYNTAX:
            %   cfg = Config.getInstance()
            %   cfg = Config.getInstance('config.json')
            %   cfg = Config.getInstance(config_struct)
            
            persistent instance
            
            if nargin == 0 && ~isempty(instance)
                % Return existing instance
                obj = instance;
            else
                % Create new instance
                if nargin == 0
                    instance = Config();
                else
                    instance = Config(source);
                end
                obj = instance;
            end
        end
        
        function cfg = getDefaultConfig()
            %GETDEFAULTCONFIG Returns default configuration structure
            
            cfg = struct();
            
            % === PATHS ===
            cfg.paths.data = 'data/';
            cfg.paths.models = 'models/';
            cfg.paths.results = 'results/';
            cfg.paths.figures = 'results/figures/';
            
            % === CROSS-VALIDATION ===
            cfg.n_folds = 10;
            cfg.n_repeats = 10;
            cfg.random_seed = 42;
            cfg.stratified = true;
            cfg.patient_level = true;  % Always use patient-level splitting
            
            % === PARALLEL PROCESSING ===
            cfg.parallel = true;
            cfg.max_workers = [];  % Use MATLAB default
            
            % === PREPROCESSING PERMUTATIONS ===
            % BSNCX notation: [Binning][Smoothing][Normalization][Correction][X]
            % Position 3 (N): 0=none, 2=vector normalization
            % Position 4 (C): 0=none, 1=1st deriv, 2=2nd deriv
            cfg.preprocessing_permutations = {
                '10000X'  % Baseline: no preprocessing
                '10200X'  % Normalization only
                '10220X'  % Normalization + 2nd derivative
                '20220X'  % Bin=2 + Norm + 2nd deriv
                '30220X'  % Bin=3 + Norm + 2nd deriv
                '11220X'  % Smooth=1 + Norm + 2nd deriv
                '21220X'  % Bin=2 + Smooth=1 + Norm + 2nd deriv
            };
            
            % === CLASSIFIERS ===
            cfg.classifiers = {'PCA-LDA', 'SVM-RBF', 'PLS-DA', 'RandomForest'};
            
            % === CLASSIFIER HYPERPARAMETERS ===
            % PCA-LDA
            cfg.pca_variance_threshold = 0.95;
            cfg.pca_max_components = 15;
            cfg.lda_delta = 0;  % Regularization
            cfg.lda_gamma = 0;
            
            % SVM-RBF
            cfg.svm_C = 1.0;
            cfg.svm_kernel_scale = 'auto';
            cfg.svm_standardize = false;  % Already done in preprocessing
            
            % PLS-DA
            cfg.pls_ncomp = 10;
            
            % Random Forest
            cfg.rf_ntrees = 100;
            cfg.rf_min_leaf_size = 1;
            
            % === OPTIMIZATION (FUTURE FEATURE) ===
            cfg.optimize_hyperparameters = false;
            cfg.optimization_max_evals = 30;
            
            % === REPORTING ===
            cfg.save_intermediate = true;
            cfg.save_models = true;
            cfg.save_predictions = true;
            cfg.figure_format = 'png';
            cfg.figure_dpi = 300;
            cfg.verbose = true;
            
            % === COST-SENSITIVE LEARNING (OPTIONAL) ===
            cfg.cost_sensitive = false;
            cfg.cost_matrix = [];  % Will be computed if cost_sensitive=true
            
            % === DATA VALIDATION ===
            cfg.handle_nan = 'error';  % 'error', 'remove', 'impute'
            cfg.min_samples_per_patient = 1;
            cfg.min_features = 10;
        end
    end
    
    methods
        function loadFromFile(obj, filename)
            %LOADFROMFILE Load configuration from JSON or MAT file
            
            [~, ~, ext] = fileparts(filename);
            
            switch lower(ext)
                case '.json'
                    % Read JSON file
                    fid = fopen(filename, 'r');
                    if fid == -1
                        error('Config:FileNotFound', ...
                            'Cannot open file: %s', filename);
                    end
                    raw = fread(fid, inf, 'char=>char')';
                    fclose(fid);
                    cfg = jsondecode(raw);
                    
                case '.mat'
                    % Load MAT file
                    loaded = load(filename);
                    if isfield(loaded, 'cfg')
                        cfg = loaded.cfg;
                    elseif isfield(loaded, 'config')
                        cfg = loaded.config;
                    else
                        error('Config:InvalidMAT', ...
                            'MAT file must contain "cfg" or "config" variable');
                    end
                    
                otherwise
                    error('Config:UnsupportedFormat', ...
                        'File format must be .json or .mat');
            end
            
            % Merge with defaults
            obj.config_struct = Config.mergeWithDefaults(cfg);
        end
        
        function loadFromStruct(obj, cfg)
            %LOADFROMSTRUCT Load configuration from struct
            
            % Merge with defaults
            obj.config_struct = Config.mergeWithDefaults(cfg);
        end
        
        function validateConfig(obj)
            %VALIDATECONFIG Validate all configuration parameters
            
            cfg = obj.config_struct;
            
            % Validate numeric parameters
            assert(cfg.n_folds > 0 && cfg.n_folds == floor(cfg.n_folds), ...
                'n_folds must be a positive integer');
            assert(cfg.n_repeats > 0 && cfg.n_repeats == floor(cfg.n_repeats), ...
                'n_repeats must be a positive integer');
            assert(isnumeric(cfg.random_seed), 'random_seed must be numeric');
            
            % Validate logical parameters
            assert(islogical(cfg.parallel) || isnumeric(cfg.parallel), ...
                'parallel must be logical');
            assert(islogical(cfg.stratified) || isnumeric(cfg.stratified), ...
                'stratified must be logical');
            
            % Validate cell arrays
            assert(iscell(cfg.preprocessing_permutations) && ...
                   ~isempty(cfg.preprocessing_permutations), ...
                'preprocessing_permutations must be a non-empty cell array');
            assert(iscell(cfg.classifiers) && ~isempty(cfg.classifiers), ...
                'classifiers must be a non-empty cell array');
            
            % Validate permutation strings
            for i = 1:length(cfg.preprocessing_permutations)
                perm = cfg.preprocessing_permutations{i};
                assert(ischar(perm) || isstring(perm), ...
                    'Permutation %d must be a string', i);
                assert(length(perm) == 6 && perm(end) == 'X', ...
                    'Permutation %d must follow BSNCX format (e.g., "10022X")', i);
            end
            
            % Validate classifier names
            valid_classifiers = {'PCA-LDA', 'SVM-RBF', 'PLS-DA', 'RandomForest'};
            for i = 1:length(cfg.classifiers)
                assert(ismember(cfg.classifiers{i}, valid_classifiers), ...
                    'Invalid classifier: %s. Must be one of: %s', ...
                    cfg.classifiers{i}, strjoin(valid_classifiers, ', '));
            end
            
            % Validate PCA parameters
            assert(cfg.pca_variance_threshold > 0 && ...
                   cfg.pca_variance_threshold <= 1, ...
                'pca_variance_threshold must be in (0, 1]');
            
            fprintf('✓ Configuration validated successfully\n');
        end
        
        function value = get(obj, field)
            %GET Get configuration value
            %
            % SYNTAX:
            %   value = cfg.get('n_folds')
            %   value = cfg.get('paths.data')
            
            % Handle nested fields (e.g., 'paths.data')
            fields = strsplit(field, '.');
            value = obj.config_struct;
            
            for i = 1:length(fields)
                if isfield(value, fields{i})
                    value = value.(fields{i});
                else
                    error('Config:FieldNotFound', ...
                        'Configuration field not found: %s', field);
                end
            end
        end
        
        function set(obj, field, value)
            %SET Set configuration value (use with caution)
            %
            % SYNTAX:
            %   cfg.set('n_folds', 5)
            
            % Handle nested fields
            fields = strsplit(field, '.');
            
            if length(fields) == 1
                obj.config_struct.(fields{1}) = value;
            else
                % Navigate to parent struct
                temp = obj.config_struct;
                for i = 1:length(fields)-1
                    if ~isfield(temp, fields{i})
                        temp.(fields{i}) = struct();
                    end
                    temp = temp.(fields{i});
                end
                temp.(fields{end}) = value;
                
                % Reconstruct from root
                obj.config_struct.(fields{1}) = temp;
            end
            
            % Re-validate after changes
            obj.validateConfig();
        end
        
        function s = getStruct(obj)
            %GETSTRUCT Get complete configuration as struct
            s = obj.config_struct;
        end
        
        function save(obj, filename)
            %SAVE Save configuration to file
            %
            % SYNTAX:
            %   cfg.save('config.json')
            %   cfg.save('config.mat')
            
            [~, ~, ext] = fileparts(filename);
            
            switch lower(ext)
                case '.json'
                    % Save as JSON
                    json_str = jsonencode(obj.config_struct);
                    fid = fopen(filename, 'w');
                    fprintf(fid, '%s', json_str);
                    fclose(fid);
                    
                case '.mat'
                    % Save as MAT
                    cfg = obj.config_struct; %#ok<NASGU>
                    save(filename, 'cfg');
                    
                otherwise
                    error('Config:UnsupportedFormat', ...
                        'File format must be .json or .mat');
            end
            
            fprintf('✓ Configuration saved: %s\n', filename);
        end
        
        function disp(obj)
            %DISP Display configuration
            fprintf('Configuration:\n');
            disp(obj.config_struct);
        end
    end
    
    methods (Static, Access = private)
        function merged = mergeWithDefaults(cfg)
            %MERGEWITHDEFAULTS Merge user config with defaults
            
            defaults = Config.getDefaultConfig();
            merged = defaults;
            
            % Recursively merge structures
            merged = Config.mergeStructs(defaults, cfg);
        end
        
        function result = mergeStructs(s1, s2)
            %MERGESTRUCTS Recursively merge two structures
            %   s1: default values
            %   s2: user-provided values (overrides s1)
            
            result = s1;
            fields = fieldnames(s2);
            
            for i = 1:length(fields)
                field = fields{i};
                
                if isfield(s1, field) && isstruct(s1.(field)) && isstruct(s2.(field))
                    % Recursively merge nested structs
                    result.(field) = Config.mergeStructs(s1.(field), s2.(field));
                else
                    % Override with user value
                    result.(field) = s2.(field);
                end
            end
        end
    end
end
