clc;
clear;

% -------------------------------------------------------------------------
% 1. LOAD MODELS, VOCABULARIES AND IDF VECTORS
% -------------------------------------------------------------------------
disp('--- PHASE 1: Loading Models, Vocabularies and IDF vectors ---');
try
    load('bow_results.mat', 'Mdl_Title', 'Mdl_Text', ...
         'vocabulary_title', 'vocabulary_text', ...
         'idf_title', 'idf_text');
    disp('✅ Models (Mdl_Title, Mdl_Text), vocabularies and IDF vectors loaded.');
catch
    error(['🔴 Error: could not load Mdl_Title, Mdl_Text, ' ...
           'vocabulary_*, idf_* from bow_results.mat.']);
end

% -------------------------------------------------------------------------
% 2. DATA LOADING AND FILTERING
% -------------------------------------------------------------------------
disp('--- PHASE 2: Loading Data (url-versions-2015-06-14.csv) ---');

fileName = 'url-versions-2015-06-14.csv';

try
    data = readtable(fileName, 'TextType', 'string');
    disp(['✅ File "', fileName, '" loaded successfully.']);
catch
    error(['🔴 Error: Could not read file ', fileName, ...
           '. Make sure it is in the same folder as this script.']);
end

% Check required columns
requiredVars = ["claimHeadline", "claimTruthiness"];
if ~all(ismember(requiredVars, string(data.Properties.VariableNames)))
    error('🔴 The file must contain "claimHeadline" and "claimTruthiness" columns.');
end

% Text column (change if the column has a different name)
textColumnName = "articleBody";   % <--- CHANGE HERE if needed
if ~ismember(textColumnName, string(data.Properties.VariableNames))
    error(['🔴 Text column "', textColumnName, '" not found in CSV. ', ...
           'Update textColumnName in this script.']);
end

% Keep only rows labelled "true" or "false"
disp('Filtering rows with claimTruthiness = "true" or "false"...');
truthiness  = data.claimTruthiness;
mask_valid  = (truthiness == "true") | (truthiness == "false");
data_valid  = data(mask_valid, :);

disp(['Total rows in file:          ', num2str(height(data))]);
disp(['Rows used for testing (true/false): ', num2str(height(data_valid))]);

if height(data_valid) == 0
    error('🔴 No rows with claimTruthiness = "true" or "false". Cannot proceed.');
end

% Map labels to "Fact" / "Fake"
labels_raw = data_valid.claimTruthiness;
Y_test     = strings(height(data_valid), 1);
Y_test(labels_raw == "true")  = "Fact";
Y_test(labels_raw == "false") = "Fake";
Y_test_str = string(Y_test);

% -------------------------------------------------------------------------
% 3. TEST ON TITLES (Mdl_Title, vocabulary_title, idf_title)
% -------------------------------------------------------------------------
disp(' ');
disp('===================================================================');
disp('TEST 1: TITLES (claimHeadline) using Mdl_Title');
disp('===================================================================');

titles = data_valid.claimHeadline;

disp('Tokenizing TITLES...');
tokens_title = cellfun(@process_text, titles, 'UniformOutput', false);

disp('Building BoW for TITLES...');
bow_test_title = create_bow_matrix(tokens_title, vocabulary_title);

% Apply TF-IDF using the IDF vector from training
X_test_title = bow_test_title .* idf_title;

disp('Predicting with Mdl_Title...');
Y_pred_title     = predict(Mdl_Title, X_test_title);
Y_pred_title_str = string(Y_pred_title);

accuracy_title = mean(Y_pred_title_str == Y_test_str);

disp('-------------------------------------------------------------------');
disp(['✅ ACCURACY on TITLES: ', num2str(accuracy_title * 100, '%.2f'), '%']);
disp('-------------------------------------------------------------------');

% Confusion matrix — Titles
figure;
confusionchart(Y_test_str, Y_pred_title_str);
title('Confusion Matrix — TITLES (url-versions-2015-06-14)');

% -------------------------------------------------------------------------
% 4. TEST ON FULL TEXTS (Mdl_Text, vocabulary_text, idf_text)
% -------------------------------------------------------------------------
disp(' ');
disp('===================================================================');
disp(['TEST 2: FULL TEXTS (', char(textColumnName), ') using Mdl_Text']);
disp('===================================================================');

texts = data_valid.(textColumnName);

disp('Tokenizing FULL TEXTS...');
tokens_text = cellfun(@process_text, texts, 'UniformOutput', false);

disp('Building BoW for FULL TEXTS...');
bow_test_text = create_bow_matrix(tokens_text, vocabulary_text);

% Apply TF-IDF using the IDF vector from training
X_test_text = bow_test_text .* idf_text;

disp('Predicting with Mdl_Text...');
Y_pred_text     = predict(Mdl_Text, X_test_text);
Y_pred_text_str = string(Y_pred_text);

accuracy_text = mean(Y_pred_text_str == Y_test_str);

disp('-------------------------------------------------------------------');
disp(['✅ ACCURACY on FULL TEXTS: ', num2str(accuracy_text * 100, '%.2f'), '%']);
disp('-------------------------------------------------------------------');

% Confusion matrix — Full texts
figure;
confusionchart(Y_test_str, Y_pred_text_str);
title(['Confusion Matrix — FULL TEXTS (', char(textColumnName), ', url-versions-2015-06-14)']);

disp('===================================================================');
disp('SCRIPT test_emergent_urlversions.m COMPLETED.');
disp('===================================================================');
