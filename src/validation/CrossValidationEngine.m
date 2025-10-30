classdef CrossValidationEngine < handle
    %CROSSVALIDATIONENGINE Patient-level cross-validation with data leakage prevention
    %
    % DESCRIPTION:
    %   Implements stratified patient-level cross-validation for evaluating
    %   preprocessing permutations and classifiers. CRITICAL: Ensures no
    %   patient appears in both training and test sets.
    %
    % USAGE:
    %   cfg = Config.getInstance();
    %   cv_engine = CrossValidationEngine(cfg);
    %   results = cv_engine.run(X, y, patient_ids);
    %
    % CRITICAL FEATURES:
    %   - Patient-level stratification (not sample-level)
    %   - Parameter isolation (preprocessing fit on train only)
    %   - Reproducible with random seed
    %   - Comprehensive results structure
    %
    % See also: Config, PreprocessingPipeline, ClassifierWrapper, MetricsCalculator
    
    properties (Access = private)
        config          % Configuration object
        cv_partitions   % Cell array of CV partition objects
        verbose         % Display progress
    end
    
    methods
        function obj = CrossValidationEngine(config, varargin)
            %CROSSVALIDATIONENGINE Constructor
            %
            % SYNTAX:
            %   cv_engine = CrossValidationEngine(cfg)
            %   cv_engine = CrossValidationEngine(cfg, 'Verbose', true)
            
            p = inputParser;
            addRequired(p, 'config');
            addParameter(p, 'Verbose', true, @islogical);
            parse(p, config, varargin{:});
            
            obj.config = config;
            obj.verbose = p.Results.Verbose;
            obj.cv_partitions = {};
        end
        
        function results = run(obj, X, y, patient_ids)
            %RUN Execute full cross-validation pipeline
            %
            % SYNTAX:
            %   results = cv_engine.run(X, y, patient_ids)
            %
            % INPUTS:
            %   X: [n_samples × n_features] feature matrix
            %   y: Categorical labels
            %   patient_ids: Patient identifiers for each sample
            %
            % OUTPUTS:
            %   results: Comprehensive results structure
            
            if obj.verbose
                fprintf('\n=== CROSS-VALIDATION ENGINE ===\n');
                fprintf('Samples: %d, Features: %d\n', size(X, 1), size(X, 2));
                fprintf('Patients: %d unique\n', length(unique(patient_ids)));
            end
            
            % Get configuration parameters
            if isstruct(obj.config)
                n_folds = obj.config.n_folds;
                n_repeats = obj.config.n_repeats;
                random_seed = obj.config.random_seed;
                permutations = obj.config.preprocessing_permutations;
                classifiers = obj.config.classifiers;
                use_parallel = obj.config.parallel;
            else
                n_folds = obj.config.get('n_folds');
                n_repeats = obj.config.get('n_repeats');
                random_seed = obj.config.get('random_seed');
                permutations = obj.config.get('preprocessing_permutations');
                classifiers = obj.config.get('classifiers');
                use_parallel = obj.config.get('parallel');
            end
            
            % Generate patient-level CV partitions
            obj.generateCVPartitions(patient_ids, y, n_folds, n_repeats, random_seed);
            
            % Initialize results structure
            results = struct();
            results.config = obj.config;
            results.n_samples = size(X, 1);
            results.n_features = size(X, 2);
            results.n_patients = length(unique(patient_ids));
            results.n_permutations = length(permutations);
            results.n_classifiers = length(classifiers);
            results.n_folds = n_folds;
            results.n_repeats = n_repeats;
            results.permutations = cell(length(permutations), 1);
            
            % Main CV loop over permutations
            if use_parallel && ~isempty(ver('parallel'))
                % Parallel execution over permutations
                if obj.verbose
                    fprintf('Running %d permutations in PARALLEL mode\n', length(permutations));
                end
                
                % Pre-allocate for parallel execution
                perm_results_array = cell(length(permutations), 1);
                parfor p = 1:length(permutations)
                    perm_results_array{p} = obj.runPermutation(X, y, patient_ids, ...
                        permutations{p}, classifiers, n_repeats, n_folds);
                end
                results.permutations = perm_results_array;
            else
                % Serial execution
                if obj.verbose
                    fprintf('Running %d permutations in SERIAL mode\n', length(permutations));
                end
                
                for p = 1:length(permutations)
                    perm_results = obj.runPermutation(X, y, patient_ids, ...
                        permutations{p}, classifiers, n_repeats, n_folds);
                    results.permutations{p} = perm_results;
                end
            end
            
            if obj.verbose
                fprintf('\n=== CROSS-VALIDATION COMPLETE ===\n');
            end
        end
    end
    
    methods (Access = private)
        function generateCVPartitions(obj, patient_ids, y, n_folds, n_repeats, random_seed)
            %GENERATECVPARTITIONS Create patient-level stratified CV partitions
            %
            % CRITICAL: Partitions patients, not samples!
            
            % Get unique patients and their labels
            unique_patients = unique(patient_ids);
            n_patients = length(unique_patients);
            
            % Assign one label per patient (majority vote)
            patient_labels = categorical(zeros(n_patients, 1));
            for i = 1:n_patients
                pid = unique_patients(i);
                patient_samples = patient_ids == pid;
                patient_y = y(patient_samples);
                % Use mode (most common label)
                patient_labels(i) = mode(patient_y);
            end
            
            % Create CV partitions for each repeat
            obj.cv_partitions = cell(n_repeats, 1);
            
            for r = 1:n_repeats
                % Set random seed for reproducibility
                rng(random_seed + r - 1);
                
                % Create stratified partition on PATIENTS
                cv_partition = cvpartition(patient_labels, 'KFold', n_folds);
                obj.cv_partitions{r} = cv_partition;
            end
            
            if obj.verbose
                fprintf('Generated %d CV partitions (%d folds each)\n', ...
                    n_repeats, n_folds);
            end
        end
        
        function perm_results = runPermutation(obj, X, y, patient_ids, ...
                                               permutation_id, classifiers, n_repeats, n_folds)
            %RUNPERMUTATION Run CV for one preprocessing permutation
            
            perm_results = struct();
            perm_results.permutation_id = permutation_id;
            perm_results.classifiers = cell(length(classifiers), 1);
            
            % Create preprocessing pipeline
            pipeline = PreprocessingPipeline(permutation_id, 'Verbose', false);
            
            % Loop over classifiers
            for c = 1:length(classifiers)
                clf_results = struct();
                clf_results.classifier_name = classifiers{c};
                clf_results.repeats = cell(n_repeats, 1);
                
                % Loop over repeats
                for r = 1:n_repeats
                    repeat_results = struct();
                    repeat_results.folds = cell(n_folds, 1);
                    
                    % Loop over folds
                    for f = 1:n_folds
                        fold_results = obj.runFold(X, y, patient_ids, ...
                            pipeline, classifiers{c}, r, f);
                        repeat_results.folds{f} = fold_results;
                    end
                    
                    clf_results.repeats{r} = repeat_results;
                end
                
                perm_results.classifiers{c} = clf_results;
            end
        end
        
        function fold_results = runFold(obj, X, y, patient_ids, ...
                                        pipeline, classifier_name, repeat_idx, fold_idx)
            %RUNFOLD Execute one CV fold
            %
            % CRITICAL: Ensures no patient overlap between train/test
            
            % Get patient-level train/test indices
            cv_partition = obj.cv_partitions{repeat_idx};
            train_patients_idx = training(cv_partition, fold_idx);
            test_patients_idx = test(cv_partition, fold_idx);
            
            % Get unique patients
            unique_patients = unique(patient_ids);
            train_patients = unique_patients(train_patients_idx);
            test_patients = unique_patients(test_patients_idx);
            
            % Map patients to samples
            train_mask = ismember(patient_ids, train_patients);
            test_mask = ismember(patient_ids, test_patients);
            
            % Split data
            X_train = X(train_mask, :);
            y_train = y(train_mask);
            pid_train = patient_ids(train_mask);
            
            X_test = X(test_mask, :);
            y_test = y(test_mask);
            pid_test = patient_ids(test_mask);
            
            % CRITICAL VALIDATION: No patient overlap
            overlap = intersect(unique(pid_train), unique(pid_test));
            if ~isempty(overlap)
                error('CrossValidationEngine:PatientLeakage', ...
                    'Patient overlap detected in fold %d, repeat %d', fold_idx, repeat_idx);
            end
            
            % Preprocessing: fit on train, transform both
            [X_train_proc, preproc_params] = pipeline.fit_transform(X_train);
            X_test_proc = pipeline.transform(X_test, preproc_params);
            
            % Train classifier
            clf = ClassifierWrapper(classifier_name, obj.config, 'Verbose', false);
            clf.train(X_train_proc, y_train);
            
            % Predict on test set
            [y_pred, scores] = clf.predict(X_test_proc);
            
            % Store results
            fold_results = struct();
            fold_results.repeat = repeat_idx;
            fold_results.fold = fold_idx;
            fold_results.n_train_samples = sum(train_mask);
            fold_results.n_test_samples = sum(test_mask);
            fold_results.n_train_patients = length(unique(pid_train));
            fold_results.n_test_patients = length(unique(pid_test));
            fold_results.y_true = y_test;
            fold_results.y_pred = y_pred;
            fold_results.scores = scores;
            fold_results.patient_ids = pid_test;
            fold_results.preprocessing_params = preproc_params;
        end
    end
end
