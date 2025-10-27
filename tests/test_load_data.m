% Simple test to load train data
fprintf('Test 1: Loading train file...\n');
try
    m = matfile('data/data_table_train.mat');
    fprintf('  Matfile opened successfully\n');
    fprintf('  Variables: %s\n', strjoin(who(m), ', '));
catch ME
    fprintf('  ERROR: %s\n', ME.message);
    return;
end

fprintf('\nTest 2: Loading with load()...\n');
try
    s = load('data/data_table_train.mat');
    fprintf('  Load successful\n');
    fprintf('  Fields: %s\n', strjoin(fieldnames(s), ', '));
catch ME
    fprintf('  ERROR: %s\n', ME.message);
    return;
end

fprintf('\nTest 3: Accessing data...\n');
try
    data = s.data_table_train;
    fprintf('  Data accessed: %d rows\n', height(data));
catch ME
    fprintf('  ERROR: %s\n', ME.message);
    return;
end

fprintf('\nAll tests passed!\n');
