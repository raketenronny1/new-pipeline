% Minimal CV test to find the error
addpath('src/meningioma_ftir_pipeline');

fprintf('=== MINIMAL CV DEBUG TEST ===\n\n');

% Load config
cfg = config();
cfg.cv.n_folds = 3;
cfg.cv.n_repeats = 1;
cfg.optimization.enabled = false;

% Load data
fprintf('Loading data...\n');
data = load_data_with_eda(cfg);
fprintf('  Train: %d samples, %d spectra\n', data.train.n_samples, data.train.total_spectra);

% Try just one classifier on one fold
fprintf('\nTesting LDA on fold 1...\n');

try
    % Create simple train/val split (first 2/3 for train, last 1/3 for val)
    n_train = floor(2/3 * data.train.n_samples);
    train_idx = 1:n_train;
    val_idx = (n_train+1):data.train.n_samples;
    
    % Extract train data
    X_train = [];
    y_train = [];
    for i = train_idx
        X_train = [X_train; data.train.spectra{i}];
        y_train = [y_train; repmat(data.train.labels(i), size(data.train.spectra{i}, 1), 1)];
    end
    
    % Extract val data
    X_val = [];
    y_val = [];
    for i = val_idx
        X_val = [X_val; data.train.spectra{i}];
        y_val = [y_val; repmat(data.train.labels(i), size(data.train.spectra{i}, 1), 1)];
    end
    
    fprintf('  Train: %d spectra\n', size(X_train, 1));
    fprintf('  Val: %d spectra\n', size(X_val, 1));
    
    % Standardize
    fprintf('  Standardizing...\n');
    X_mean = mean(X_train, 1);
    X_std = std(X_train, 0, 1);
    X_std(X_std == 0) = 1;
    
    X_train_std = (X_train - X_mean) ./ X_std;
    X_val_std = (X_val - X_mean) ./ X_std;
    
    % Apply PCA
    fprintf('  Applying PCA transform...\n');
    X_train_pca = (X_train_std - data.pca_model.X_mean) * data.pca_model.coeff;
    X_val_pca = (X_val_std - data.pca_model.X_mean) * data.pca_model.coeff;
    
    fprintf('  PCA features: %d\n', size(X_train_pca, 2));
    
    % Train LDA
    fprintf('  Training LDA...\n');
    mdl = fitcdiscr(X_train_pca, y_train);
    
    % Predict
    fprintf('  Predicting...\n');
    [y_pred, scores] = predict(mdl, X_val_pca);
    
    % Metrics
    acc = sum(y_pred == y_val) / length(y_val);
    fprintf('  Accuracy: %.2f%%\n', 100*acc);
    
    fprintf('\n✓ Test passed!\n');
    
catch ME
    fprintf('\n✗ ERROR: %s\n', ME.message);
    for i = 1:min(5, length(ME.stack))
        fprintf('  at %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
end
