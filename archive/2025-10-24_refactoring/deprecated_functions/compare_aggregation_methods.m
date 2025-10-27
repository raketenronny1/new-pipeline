%% COMPARE AGGREGATION METHODS
% Compare majority vote vs mean spectra prediction approaches
%
% This script tests two sample-level aggregation methods:
% 1. Majority Vote: Train on individual spectra, aggregate predictions
% 2. Mean Spectra: Average spectra per sample first, then predict

function compare_aggregation_methods()
    fprintf('\n=== COMPARING AGGREGATION METHODS ===\n\n');
    
    % Load configuration and data
    addpath('src/meningioma_ftir_pipeline');
    cfg = config();
    data = load_data_direct(cfg);
    
    %% Method 1: Majority Vote (Current Approach)
    fprintf('METHOD 1: MAJORITY VOTE\n');
    fprintf('  Training on individual spectra, aggregating predictions\n\n');
    
    % Extract all spectra
    [X_train, y_train, train_sample_map] = extract_all_spectra_with_map(data.train, 1:data.train.n_samples);
    [X_test, y_test, test_sample_map] = extract_all_spectra_with_map(data.test, 1:data.test.n_samples);
    
    % Standardize
    [X_train_std, std_params] = standardize_spectra_train(X_train);
    X_test_std = standardize_spectra_test(X_test, std_params);
    
    % Train PLSDA
    fprintf('  Training PLSDA on %d training spectra...\n', size(X_train_std, 1));
    mdl_vote = fitclinear(X_train_std, y_train, 'Learner', 'logistic', ...
                          'ObservationsIn', 'rows');
    
    % Predict - get PROBABILITIES (posteriors)
    [spectrum_preds, spectrum_scores] = predict(mdl_vote, X_test_std);
    
    % Method 1a: Aggregate via MAJORITY VOTE
    sample_preds_vote = aggregate_to_samples_vote(spectrum_preds, test_sample_map, data.test.n_samples);
    metrics_vote = compute_metrics(data.test.labels, sample_preds_vote);
    
    fprintf('  Results (Majority Vote - Hard Predictions):\n');
    fprintf('    Accuracy:    %.1f%%\n', metrics_vote.accuracy * 100);
    fprintf('    Sensitivity: %.1f%%\n', metrics_vote.sensitivity * 100);
    fprintf('    Specificity: %.1f%%\n', metrics_vote.specificity * 100);
    
    %% Method 2: Mean Posteriors (Average Probabilities)
    fprintf('\nMETHOD 2: MEAN POSTERIORS\n');
    fprintf('  Training on individual spectra (same as Method 1)\n');
    fprintf('  Aggregating via MEAN of prediction probabilities\n\n');
    
    % Aggregate via mean of posteriors
    % spectrum_scores is Nx2: [P(WHO-1), P(WHO-3)]
    sample_preds_mean_post = aggregate_to_samples_mean_posterior(spectrum_scores, test_sample_map, data.test.n_samples);
    metrics_mean_post = compute_metrics(data.test.labels, sample_preds_mean_post);
    
    fprintf('  Results (Mean Posteriors - Soft Probabilities):\n');
    fprintf('    Accuracy:    %.1f%%\n', metrics_mean_post.accuracy * 100);
    fprintf('    Sensitivity: %.1f%%\n', metrics_mean_post.sensitivity * 100);
    fprintf('    Specificity: %.1f%%\n', metrics_mean_post.specificity * 100);
    
    %% Comparison
    fprintf('\n=== COMPARISON ===\n');
    fprintf('┌──────────────────────┬──────────────┬──────────────┬────────────┐\n');
    fprintf('│ Method               │   Accuracy   │ Sensitivity  │ Specificity│\n');
    fprintf('├──────────────────────┼──────────────┼──────────────┼────────────┤\n');
    fprintf('│ Majority Vote        │    %.1f%%     │    %.1f%%     │   %.1f%%    │\n', ...
            metrics_vote.accuracy*100, metrics_vote.sensitivity*100, metrics_vote.specificity*100);
    fprintf('│ Mean Posteriors      │    %.1f%%     │    %.1f%%     │   %.1f%%    │\n', ...
            metrics_mean_post.accuracy*100, metrics_mean_post.sensitivity*100, metrics_mean_post.specificity*100);
    fprintf('└──────────────────────┴──────────────┴──────────────┴────────────┘\n');
    
    % Calculate improvement
    acc_diff = (metrics_mean_post.accuracy - metrics_vote.accuracy) * 100;
    sens_diff = (metrics_mean_post.sensitivity - metrics_vote.sensitivity) * 100;
    spec_diff = (metrics_mean_post.specificity - metrics_vote.specificity) * 100;
    
    fprintf('\nDifference (Mean Posteriors - Majority Vote):\n');
    fprintf('  Accuracy:    %+.1f%%\n', acc_diff);
    fprintf('  Sensitivity: %+.1f%%\n', sens_diff);
    fprintf('  Specificity: %+.1f%%\n', spec_diff);
    
    if metrics_mean_post.accuracy > metrics_vote.accuracy
        fprintf('\n✓ Mean posteriors method performs BETTER\n');
    elseif metrics_mean_post.accuracy < metrics_vote.accuracy
        fprintf('\n✗ Majority vote method performs BETTER\n');
    else
        fprintf('\n= Both methods perform EQUALLY\n');
    end
end

%% Helper Functions

function [X, y, sample_map] = extract_all_spectra_with_map(data, sample_indices)
    X = [];
    y = [];
    sample_map = [];
    for i = 1:length(sample_indices)
        idx = sample_indices(i);
        spectra = data.spectra{idx};
        n_spec = size(spectra, 1);
        labels = repmat(data.labels(idx), n_spec, 1);
        X = [X; spectra];
        y = [y; labels];
        sample_map = [sample_map; repmat(i, n_spec, 1)];
    end
end

function [X_std, params] = standardize_spectra_train(X)
    mu = mean(X, 1);
    sigma = std(X, 0, 1);
    sigma(sigma == 0) = 1;
    X_std = (X - mu) ./ sigma;
    params = struct('mu', mu, 'sigma', sigma);
end

function X_std = standardize_spectra_test(X, params)
    X_std = (X - params.mu) ./ params.sigma;
end

function sample_preds = aggregate_to_samples_vote(spectrum_preds, sample_map, n_samples)
    % Majority vote on hard predictions
    sample_preds = zeros(n_samples, 1);
    for i = 1:n_samples
        sample_spectra_preds = spectrum_preds(sample_map == i);
        sample_preds(i) = mode(sample_spectra_preds);
    end
end

function sample_preds = aggregate_to_samples_mean_posterior(spectrum_scores, sample_map, n_samples)
    % Average posteriors (probabilities) then threshold
    % spectrum_scores is Nx2: [P(WHO-1), P(WHO-3)]
    sample_preds = zeros(n_samples, 1);
    for i = 1:n_samples
        sample_scores = spectrum_scores(sample_map == i, :);
        % Average probabilities across all spectra in this sample
        mean_prob = mean(sample_scores, 1);
        % mean_prob is [P(WHO-1), P(WHO-3)]
        % Classify based on which probability is higher
        [~, pred_class] = max(mean_prob);
        % pred_class is 1 or 2, map to WHO grades (1 or 3)
        if pred_class == 1
            sample_preds(i) = 1;  % WHO-1
        else
            sample_preds(i) = 3;  % WHO-3
        end
    end
end

function metrics = compute_metrics(y_true, y_pred)
    cm = confusionmat(y_true, y_pred);
    
    % For binary: [TN FP; FN TP]
    TN = cm(1,1);
    FP = cm(1,2);
    FN = cm(2,1);
    TP = cm(2,2);
    
    metrics.accuracy = (TP + TN) / (TP + TN + FP + FN);
    metrics.sensitivity = TP / (TP + FN);
    metrics.specificity = TN / (TN + FP);
    metrics.precision = TP / (TP + FP);
end
