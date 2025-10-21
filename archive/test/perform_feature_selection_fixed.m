function perform_feature_selection_fixed(cfg)
    %% Load Preprocessed Data
    fprintf('Loading preprocessed data...\n');
    load(fullfile(cfg.paths.results, 'preprocessed_data.mat'), 'trainingData');
    
    % Debug y field format
    fprintf('DEBUG: trainingData.y class: %s\n', class(trainingData.y));
    fprintf('DEBUG: trainingData.y size: [%s]\n', mat2str(size(trainingData.y)));

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

    %% Create Visualizations
    % Scree plot and cumulative variance
    figure('Position', [100, 100, 1000, 400]);

    % Scree plot
    subplot(1,2,1);
    plot(1:min(20, length(explained)), explained(1:min(20, length(explained))), 'bo-');
    xlabel('Principal Component');
    ylabel('Variance Explained (%)');
    title('Scree Plot');
    grid on;

    % Cumulative variance
    subplot(1,2,2);
    plot(1:min(20, length(explained)), cumsum(explained(1:min(20, length(explained)))), 'ro-');
    hold on;
    yline(95, 'k--', '95%');
    yline(99, 'k--', '99%');
    xlabel('Principal Component');
    ylabel('Cumulative Variance (%)');
    title('Cumulative Variance');
    grid on;

    % Save plot
    saveas(gcf, fullfile(cfg.paths.results, 'pca_variance_explained.png'));

    %% Create 2D PCA Plot Colored by WHO Grade
    figure('Position', [100, 100, 800, 600]);

    % Debug class distribution
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

    % Plot WHO-1 samples
    scatter(score(who1_indices, 1), ...
            score(who1_indices, 2), ...
            100, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;

    % Plot WHO-3 samples
    scatter(score(who3_indices, 1), ...
            score(who3_indices, 2), ...
            100, 'r', 'filled', 'MarkerFaceAlpha', 0.6);

    xlabel('PC1');
    ylabel('PC2');
    legend({'WHO-1', 'WHO-3'}, 'Location', 'best');
    title('Training Data in PCA Space');
    grid on;

    % Save plot
    saveas(gcf, fullfile(cfg.paths.results, 'pca_training_space.png'));

    pca_model = struct();
    pca_model.coeff = coeff;  % PC loadings [n_wavenumbers × n_components]
    pca_model.mu = mu;  % Training data mean [1 × n_wavenumbers]
    pca_model.latent = latent;  % Eigenvalues
    pca_model.explained = explained;  % Variance explained per PC
    pca_model.n_components = n_components;  % Number of components to keep
    
    % Save transformed training data (project to PC space)
    % CRITICAL FIX: Save the variable with the correct name that is expected by run_cross_validation.m
    X_train_pca_reduced = X_train_pca(:, 1:n_components);
    
    % Save the variable with the EXACT name expected by run_cross_validation.m
    % This is the critical fix
    save(fullfile(cfg.paths.results, 'X_train_pca.mat'), 'X_train_pca_reduced', 'X_train_pca');
    
    save(fullfile(cfg.paths.models, 'pca_model.mat'), 'pca_model');
    
    % Also save full PC scores for further analysis if needed
    save(fullfile(cfg.paths.results, 'full_pca_results.mat'), ...
         'X_train_pca', 'coeff', 'mu', 'latent', 'explained');
end