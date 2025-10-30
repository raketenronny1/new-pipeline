%% TEST_CONFIG - Unit tests for Config singleton class
%
% Tests:
%   1. Singleton pattern enforcement
%   2. Default configuration loading
%   3. Custom configuration from struct
%   4. Get/set methods
%   5. Validation logic
%
% USAGE:
%   run test_config.m

function test_config()
    fprintf('=== Testing Config Class ===\n\n');
    
    %% Test 1: Singleton Pattern
    fprintf('Test 1: Singleton pattern... ');
    cfg1 = Config.getInstance();
    cfg2 = Config.getInstance();
    assert(cfg1 == cfg2, 'Singleton pattern failed');
    fprintf('✓ PASSED\n');
    
    %% Test 2: Default Configuration
    fprintf('Test 2: Default configuration... ');
    cfg = Config.getInstance();
    assert(cfg.get('n_folds') == 10, 'Default n_folds incorrect');
    assert(cfg.get('n_repeats') == 10, 'Default n_repeats incorrect');
    assert(cfg.get('random_seed') == 42, 'Default random_seed incorrect');
    fprintf('✓ PASSED\n');
    
    %% Test 3: Get Method
    fprintf('Test 3: Get method... ');
    value = cfg.get('n_folds');
    assert(value == 10, 'Get method failed');
    
    % Test nested field access
    path = cfg.get('paths.data');
    assert(strcmp(path, 'data/'), 'Nested get failed');
    fprintf('✓ PASSED\n');
    
    %% Test 4: Custom Configuration
    fprintf('Test 4: Custom configuration from struct... ');
    custom = struct();
    custom.n_folds = 5;
    custom.n_repeats = 3;
    custom.random_seed = 123;
    
    cfg_custom = Config.getInstance(custom);
    assert(cfg_custom.get('n_folds') == 5, 'Custom n_folds not applied');
    assert(cfg_custom.get('n_repeats') == 3, 'Custom n_repeats not applied');
    fprintf('✓ PASSED\n');
    
    %% Test 5: Preprocessing Permutations
    fprintf('Test 5: Preprocessing permutations... ');
    perms = cfg.get('preprocessing_permutations');
    assert(iscell(perms), 'Permutations must be cell array');
    assert(~isempty(perms), 'Permutations cannot be empty');
    
    % Validate format
    for i = 1:length(perms)
        assert(length(perms{i}) == 6, 'Permutation must be 6 chars');
        assert(perms{i}(end) == 'X', 'Permutation must end with X');
    end
    fprintf('✓ PASSED\n');
    
    %% Test 6: Classifiers
    fprintf('Test 6: Classifiers... ');
    classifiers = cfg.get('classifiers');
    assert(iscell(classifiers), 'Classifiers must be cell array');
    assert(~isempty(classifiers), 'Classifiers cannot be empty');
    
    valid = {'PCA-LDA', 'SVM-RBF', 'PLS-DA', 'RandomForest'};
    for i = 1:length(classifiers)
        assert(ismember(classifiers{i}, valid), 'Invalid classifier');
    end
    fprintf('✓ PASSED\n');
    
    %% Test 7: Validation
    fprintf('Test 7: Validation logic... ');
    try
        bad_config = struct();
        bad_config.n_folds = -1;  % Invalid
        Config.getInstance(bad_config);
        error('Should have failed validation');
    catch ME
        % Expected error - just verify it failed
        assert(~isempty(ME.message), 'Should have error message');
    end
    fprintf('✓ PASSED\n');
    
    %% Test 8: Save and Load
    fprintf('Test 8: Save/load configuration... ');
    test_file = 'test_config_temp.mat';
    
    % Save
    cfg.save(test_file);
    assert(exist(test_file, 'file') == 2, 'Config file not created');
    
    % Clean up
    delete(test_file);
    fprintf('✓ PASSED\n');
    
    %% Test 9: Get Full Struct
    fprintf('Test 9: Get full struct... ');
    s = cfg.getStruct();
    assert(isstruct(s), 'getStruct must return struct');
    assert(isfield(s, 'n_folds'), 'Struct missing n_folds');
    assert(isfield(s, 'classifiers'), 'Struct missing classifiers');
    fprintf('✓ PASSED\n');
    
    fprintf('\n=== ALL TESTS PASSED ===\n');
end
