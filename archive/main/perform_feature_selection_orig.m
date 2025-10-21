
%% PHASE 2: FEATURE SELECTION ON TRAINING SET ONLY
% This script performs PCA-based feature selection using only the training data

function perform_feature_selection(cfg)
        %% Load Preprocessed Data
        fprintf('Loading preprocessed data...\n');
        load(fullfile(cfg.paths.results, 'preprocessed_data.mat'), 'trainingData');

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

% Handle labels based on their type (categorical, string, or numeric)
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
pca_model.n_components = n_components;  % Number selected
save('models/meningioma_ftir_pipeline/pca_model.mat', 'pca_model');

        %% Save PCA Model
        fprintf('Saving PCA model...\n');
        pca_model = struct();
        pca_model.coeff = coeff;  % PC loadings [n_wavenumbers × n_components]
        pca_model.mu = mu;  % Training data mean [1 × n_wavenumbers]
        pca_model.latent = latent;  % Eigenvalues
        pca_model.explained = explained;  % Variance explained per PC
        pca_model.n_components = n_components;  % Number selected
        save(fullfile(cfg.paths.models, 'pca_model.mat'), 'pca_model');

%% Transform Training Data
save('results/meningioma_ftir_pipeline/X_train_pca.mat', 'X_train_pca');

        X_train_pca = score(:, 1:n_components);
        save(fullfile(cfg.paths.results, 'X_train_pca.mat'), 'X_train_pca');

        fprintf('Feature selection complete.\n');
end