function perform_feature_selection(cfg)
    %% Load Preprocessed Data
    fprintf('Loading preprocessed data...\n');
    load(fullfile(cfg.paths.results, 'preprocessed_data.mat'), 'trainingData');
    
    % Debug y field format
    fprintf('DEBUG: Fields in trainingData: %s \n', strjoin(fieldnames(trainingData), ' '));
    fprintf('DEBUG: trainingData.y class: %s\n', class(trainingData.y));
    if iscategorical(trainingData.y)
        fprintf('DEBUG: trainingData.y is categorical with %d unique values\n', numel(categories(trainingData.y)));
        fprintf('DEBUG: trainingData.y categories: %s\n', strjoin(string(categories(trainingData.y)), ', '));
    end

    %% Compute PCA on Training Data
    fprintf('Computing PCA on training data...\n');

    % Check data validity
    if any(isnan(trainingData.X(:)) | isinf(trainingData.X(:)))
        error('Training data contains NaN/Inf values before PCA');
    end

    % Center and scale the data
    X_centered = trainingData.X - mean(trainingData.X);
    X_scaled = X_centered ./ std(trainingData.X);

    % Compute PCA manually for better control
    [U, S, V] = svd(X_scaled, 'econ');
    latent = diag(S).^2;
    explained = 100 * latent / sum(latent);
    coeff = V;
    score = X_scaled * V;
    mu = mean(trainingData.X);

    % Create transformed training data
    X_train_pca = score;

    %% Select Number of Components
    % Strategy 1: Cumulative variance threshold (95%)
    variance_threshold = 0.95;
    n_components = find(cumsum(explained) >= variance_threshold * 100, 1);
    if isempty(n_components)
        n_components = size(X_train_pca, 2);
        warning('Using all %d components as variance threshold not met', n_components);
    end

    fprintf('Selected %d PCs explaining %.1f%% variance\n', ...
            n_components, sum(explained(1:n_components)));

    %% Skip Visualizations in Batch Mode
    % Instead of creating figures, just log what would be created
    fprintf('INFO: Skipping figure generation in batch mode\n');
    fprintf('INFO: Would create scree plot showing variance per PC\n');
    fprintf('INFO: Would create cumulative variance plot (threshold: %.1f%%)\n', variance_threshold * 100);
    fprintf('INFO: Would create 2D PCA scatter plot by WHO grade\n');

    %% Prepare PCA Model Structure
    % Create a PCA model for reuse in later steps
    pca_model = struct();
    pca_model.coeff = coeff;
    pca_model.explained = explained;
    pca_model.mu = mu;
    pca_model.n_components = n_components;
    
    % Reduce to selected components only
    X_train_pca = X_train_pca(:, 1:n_components);

    %% Debug class distribution
    if iscategorical(trainingData.y)
        who1_indices = trainingData.y == 'WHO-1' | double(trainingData.y) == 1;
        who3_indices = trainingData.y == 'WHO-3' | double(trainingData.y) == 3;
    elseif isstring(trainingData.y) || iscell(trainingData.y)
        who1_indices = strcmp(trainingData.y, 'WHO-1') | strcmp(trainingData.y, '1');
        who3_indices = strcmp(trainingData.y, 'WHO-3') | strcmp(trainingData.y, '3');
    else
        % Assume numeric (1 for WHO-1, 3 for WHO-3)
        who1_indices = trainingData.y == 1;
        who3_indices = trainingData.y == 3;
    end
    
    fprintf('DEBUG: WHO-1 samples: %d, WHO-3 samples: %d\n', ...
            sum(who1_indices), sum(who3_indices));

    %% Save Results
    save(fullfile(cfg.paths.models, 'pca_model.mat'), 'pca_model');
    save(fullfile(cfg.paths.results, 'X_train_pca.mat'), 'X_train_pca');
    save(fullfile(cfg.paths.results, 'full_pca_results.mat'), 'coeff', 'score', 'explained', 'mu');

    % Also save a copy in test folder
    if contains(cfg.paths.results, 'test')
        fprintf('Saving additional copy in test folder\n');
        test_folder = fileparts(cfg.paths.results);
        save(fullfile(test_folder, 'X_train_pca.mat'), 'X_train_pca');
    end
end