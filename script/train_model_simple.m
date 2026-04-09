% =========================================================================
% FULL PIPELINE: 10-FOLD CROSS-VALIDATION + ROC CURVES + FINAL TRAINING
% =========================================================================
clear; clc; close all;

% Set random seed for reproducibility
rng('default');
K_FOLDS    = 10;
CLASS_NAMES = ["Fact", "Fake"];

% -------------------------------------------------------------------------
% 0. COLORMAP DEFINITION (for confusion matrix plots)
% -------------------------------------------------------------------------
color_low  = [253, 227, 212] / 255;
color_mid  = [173, 221, 142] / 255;
color_high = [ 49, 163,  84] / 255;
n_colors   = 256;
custom_colormap = zeros(n_colors, 3);
custom_colormap(1:n_colors/2, 1)        = linspace(color_low(1),  color_mid(1),  n_colors/2);
custom_colormap(1:n_colors/2, 2)        = linspace(color_low(2),  color_mid(2),  n_colors/2);
custom_colormap(1:n_colors/2, 3)        = linspace(color_low(3),  color_mid(3),  n_colors/2);
custom_colormap(n_colors/2+1:end, 1)    = linspace(color_mid(1),  color_high(1), n_colors/2);
custom_colormap(n_colors/2+1:end, 2)    = linspace(color_mid(2),  color_high(2), n_colors/2);
custom_colormap(n_colors/2+1:end, 3)    = linspace(color_mid(3),  color_high(3), n_colors/2);

% -------------------------------------------------------------------------
% 1. DATA LOADING
% -------------------------------------------------------------------------
disp('--- PHASE 1: Loading BoW Data ---');
try
    load('bow_results.mat', 'bow_textfact', 'bow_textfake', ...
         'bow_titlefact', 'bow_titlefake');
    disp('✅ Data loaded.');
catch
    error('🔴 Error: bow_results.mat not found. Run nuovo.m first.');
end

% =========================================================================
% ANALYSIS 1: CROSS-VALIDATION ON FULL TEXTS + ROC DATA
% =========================================================================
disp(' ');
disp('=== STARTING CV: FULL TEXTS ===');

X_text     = [bow_textfact; bow_textfake];
Y_text_str = [repmat("Fact", height(bow_textfact), 1); ...
              repmat("Fake", height(bow_textfake), 1)];

disp('Computing TF-IDF for texts...');
[num_docs_text, ~] = size(X_text);
df_text            = sum(X_text > 0, 1);
idf_text           = log((num_docs_text + 1) ./ (df_text + 1));
X_text_tfidf       = X_text .* idf_text;

Y_text_cat       = categorical(Y_text_str);
cv_indices       = cvpartition(Y_text_cat, 'KFold', K_FOLDS);
accuracies_text  = zeros(K_FOLDS, 1);
conf_matrices_text = zeros(2, 2, K_FOLDS);

% Accumulators for global ROC curve
all_scores_text = [];
all_labels_text = [];

tic;
for k = 1:K_FOLDS
    train_idx = training(cv_indices, k);
    test_idx  = test(cv_indices, k);

    Mdl_fold = fitclinear(X_text_tfidf(train_idx, :), Y_text_str(train_idx, :), ...
        'Learner', 'logistic', 'Regularization', 'lasso', 'Solver', 'sparsa');

    % Predict and collect scores for ROC
    [Y_pred, scores] = predict(Mdl_fold, X_text_tfidf(test_idx, :));

    % Identify which column corresponds to the positive class ("Fake")
    pos_class_idx = find(strcmp(Mdl_fold.ClassNames, 'Fake'));
    if isempty(pos_class_idx), pos_class_idx = 2; end

    all_scores_text = [all_scores_text; scores(:, pos_class_idx)];
    all_labels_text = [all_labels_text; Y_text_str(test_idx, :)];

    accuracies_text(k)          = mean(Y_pred == Y_text_str(test_idx, :));
    conf_matrices_text(:,:,k)   = confusionmat(Y_text_str(test_idx, :), Y_pred, 'Order', CLASS_NAMES);
end
toc;
disp(['✅ CV (Texts) complete. Mean Accuracy: ', num2str(mean(accuracies_text)*100, '%.2f'), '%']);

% --- CONFUSION MATRIX PLOT — TEXTS ---
total_conf_mat_text = sum(conf_matrices_text, 3);
figure('Color', 'w', 'Name', 'Confusion Matrix Text');
cm_text = confusionchart(total_conf_mat_text, CLASS_NAMES);
title('Total CV Confusion Matrix (FULL TEXTS)');
colormap(cm_text.Parent, custom_colormap);
cm_text.FontColor = 'k';

% =========================================================================
% ANALYSIS 2: CROSS-VALIDATION ON TITLES + ROC DATA
% =========================================================================
disp(' ');
disp('=== STARTING CV: TITLES ===');

X_title     = [bow_titlefact; bow_titlefake];
Y_title_str = [repmat("Fact", height(bow_titlefact), 1); ...
               repmat("Fake", height(bow_titlefake), 1)];

disp('Computing TF-IDF for titles...');
[num_docs_title, ~] = size(X_title);
df_title            = sum(X_title > 0, 1);
idf_title           = log((num_docs_title + 1) ./ (df_title + 1));
X_title_tfidf       = X_title .* idf_title;

Y_title_cat         = categorical(Y_title_str);
cv_indices_title    = cvpartition(Y_title_cat, 'KFold', K_FOLDS);
accuracies_title    = zeros(K_FOLDS, 1);
conf_matrices_title = zeros(2, 2, K_FOLDS);

all_scores_title = [];
all_labels_title = [];

tic;
for k = 1:K_FOLDS
    train_idx = training(cv_indices_title, k);
    test_idx  = test(cv_indices_title, k);

    Mdl_fold = fitclinear(X_title_tfidf(train_idx, :), Y_title_str(train_idx, :), ...
        'Learner', 'logistic', 'Regularization', 'lasso', 'Solver', 'sparsa');

    [Y_pred, scores] = predict(Mdl_fold, X_title_tfidf(test_idx, :));

    pos_class_idx = find(strcmp(Mdl_fold.ClassNames, 'Fake'));
    if isempty(pos_class_idx), pos_class_idx = 2; end

    all_scores_title = [all_scores_title; scores(:, pos_class_idx)];
    all_labels_title = [all_labels_title; Y_title_str(test_idx, :)];

    accuracies_title(k)          = mean(Y_pred == Y_title_str(test_idx, :));
    conf_matrices_title(:,:,k)   = confusionmat(Y_title_str(test_idx, :), Y_pred, 'Order', CLASS_NAMES);
end
toc;
disp(['✅ CV (Titles) complete. Mean Accuracy: ', num2str(mean(accuracies_title)*100, '%.2f'), '%']);

% --- CONFUSION MATRIX PLOT — TITLES ---
total_conf_mat_title = sum(conf_matrices_title, 3);
figure('Color', 'w', 'Name', 'Confusion Matrix Title');
cm_title = confusionchart(total_conf_mat_title, CLASS_NAMES);
title('Total CV Confusion Matrix (TITLES)');
colormap(cm_title.Parent, custom_colormap);
cm_title.FontColor = 'k';

% =========================================================================
% PHASE 3: ROC CURVES (COMPARATIVE PLOT)
% =========================================================================
disp(' ');
disp('=== GENERATING ROC CURVES ===');

[Xroc_text,  Yroc_text,  ~, AUC_text]  = perfcurve(all_labels_text,  all_scores_text,  'Fake');
[Xroc_title, Yroc_title, ~, AUC_title] = perfcurve(all_labels_title, all_scores_title, 'Fake');

figure('Color', 'w', 'Name', 'ROC Curves Comparison');
hold on;

% Random chance baseline
plot([0 1], [0 1], 'k--', 'LineWidth', 1.5, 'DisplayName', 'Random Chance');

% Titles ROC (orange)
plot(Xroc_title, Yroc_title, 'Color', [0.85 0.325 0.098], 'LineWidth', 2.5, ...
    'DisplayName', ['TITLES (AUC = ' num2str(AUC_title, '%.4f') ')']);

% Texts ROC (blue)
plot(Xroc_text, Yroc_text, 'Color', [0 0.447 0.741], 'LineWidth', 2.5, ...
    'DisplayName', ['FULL TEXTS (AUC = ' num2str(AUC_text, '%.4f') ')']);

xlabel('False Positive Rate (1 - Specificity)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('True Positive Rate (Sensitivity)',       'FontSize', 12, 'FontWeight', 'bold');
title('ROC Curve Comparison: Full Texts vs Titles', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'southeast', 'FontSize', 11);
grid on; box on; axis square;
xlim([0 1]); ylim([0 1]);
hold off;

disp(['AUC Full Texts: ', num2str(AUC_text)]);
disp(['AUC Titles:     ', num2str(AUC_title)]);

% =========================================================================
% PHASE 4: FINAL MODEL TRAINING AND SAVING
% =========================================================================
disp(' ');
disp('=== FINAL MODEL TRAINING AND SAVING ===');
opts_tolerance = 1e-6;
opts_iter      = 1000;

disp('Training final Mdl_Text...');
Mdl_Text = fitclinear(X_text_tfidf, Y_text_str, ...
    'Learner', 'logistic', 'Regularization', 'lasso', 'Solver', 'sparsa', ...
    'GradientTolerance', opts_tolerance, 'IterationLimit', opts_iter);

disp('Training final Mdl_Title...');
Mdl_Title = fitclinear(X_title_tfidf, Y_title_str, ...
    'Learner', 'logistic', 'Regularization', 'lasso', 'Solver', 'sparsa', ...
    'GradientTolerance', opts_tolerance, 'IterationLimit', opts_iter);

save('bow_results.mat', 'Mdl_Text', 'Mdl_Title', 'idf_text', 'idf_title', '-append');
disp('✅ SCRIPT COMPLETED SUCCESSFULLY.');
