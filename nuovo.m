clear; clc;
% -------------------------------------------------------------------------
% 1. DATA LOADING (For PHASES 2-4: Vocabulary Construction)
% -------------------------------------------------------------------------
disp('--- PHASE 1: Data Loading ---');
try
    data_true = readtable('True.csv', 'TextType', 'string');
    data_fake = readtable('Fake.csv', 'TextType', 'string');
    disp('✅ Data loaded successfully from True.csv and Fake.csv.');
catch ME
    if strcmp(ME.identifier, 'MATLAB:readtable:FileOpenError')
        error('🔴 Error: Make sure True.csv and Fake.csv are in the current MATLAB folder.');
    else
        rethrow(ME);
    end
end
% -------------------------------------------------------------------------
% 2. PRE-ALLOCATION AND TOKENIZATION WITH SAFETY CHECKS
% -------------------------------------------------------------------------
disp('--- PHASE 2: Full Tokenization ---');
N_true = height(data_true);
N_fake = height(data_fake);
textfact  = cell(N_true, 1);
textfake  = cell(N_fake, 1);
titlefact = cell(N_true, 1);
titlefake = cell(N_fake, 1);
disp('Starting tokenization: True/Fact articles...');
for i = 1:N_true
    if mod(i, 1000) == 0
        disp(['  > Tokenization True Progress: Article ', num2str(i), ' of ', num2str(N_true)]);
    end
    current_text  = data_true.text(i);
    current_title = data_true.title(i);
    if strlength(current_text) > 0
        textfact{i} = process_text(current_text);
    else
        textfact{i} = {};
    end
    if strlength(current_title) > 0
        titlefact{i} = process_text(current_title);
    else
        titlefact{i} = {};
    end
end
disp('Starting tokenization: Fake articles...');
for i = 1:N_fake
    if mod(i, 1000) == 0
        disp(['  > Tokenization Fake Progress: Article ', num2str(i), ' of ', num2str(N_fake)]);
    end
    current_text  = data_fake.text(i);
    current_title = data_fake.title(i);
    if strlength(current_text) > 0
        textfake{i} = process_text(current_text);
    else
        textfake{i} = {};
    end
    if strlength(current_title) > 0
        titlefake{i} = process_text(current_title);
    else
        titlefake{i} = {};
    end
end
disp('✅ Tokenization complete.');
% -------------------------------------------------------------------------
% 3. VOCABULARY CREATION AND SORTING
% -------------------------------------------------------------------------
disp('--- PHASE 3: Vocabulary Creation and Sorting ---');
all_text_tokens  = [textfact; textfake];
vocabulary_text  = local_sort_vocabulary(all_text_tokens);
disp(['✅ Text vocabulary created and sorted. Initial size: ', num2str(length(vocabulary_text))]);
all_title_tokens = [titlefact; titlefake];
vocabulary_title = local_sort_vocabulary(all_title_tokens);
disp(['✅ Title vocabulary created and sorted. Initial size: ', num2str(length(vocabulary_title))]);
% -------------------------------------------------------------------------
% 4. VOCABULARY FILTERING (REMOVE SINGLETON TOKENS)
% -------------------------------------------------------------------------
disp('--- PHASE 4: Vocabulary Filtering (Remove singleton tokens) ---');
[vocabulary_text_filtered, vocabulary_title_filtered] = filter_vocabulary_by_frequency(...
    textfact, textfake, titlefact, titlefake, ...
    vocabulary_text, vocabulary_title);
vocabulary_text  = vocabulary_text_filtered;
vocabulary_title = vocabulary_title_filtered;
disp('✅ Singleton tokens (frequency = 1) removed from vocabularies.');
disp(['New text vocabulary size:  ', num2str(length(vocabulary_text))]);
disp(['New title vocabulary size: ', num2str(length(vocabulary_title))]);

% --- RAM OPTIMIZATION: clear intermediate token variables ---
disp('Clearing intermediate token variables to free RAM...');
clear textfact textfake titlefact titlefake ...
      all_text_tokens all_title_tokens ...
      vocabulary_text_filtered vocabulary_title_filtered ...
      data_true data_fake current_text current_title;

% -------------------------------------------------------------------------
% 5. BAG-OF-WORDS MATRIX CONSTRUCTION (FULL AND OPTIMIZED)
% -------------------------------------------------------------------------
disp('--- PHASE 5: BoW Matrix Construction (FULL AND OPTIMIZED) ---');
disp('Building BoW: Text Fact...');
data_true   = readtable('True.csv', 'TextType', 'string');
temp_tokens = cellfun(@process_text, data_true.text, 'UniformOutput', false);
bow_textfact = create_bow_matrix(temp_tokens, vocabulary_text);
clear temp_tokens;
disp(['✅ BoW matrix (Text Fact) created. Size: ', num2str(N_true), ' x ', num2str(length(vocabulary_text))]);

disp('Building BoW: Title Fact...');
temp_tokens  = cellfun(@process_text, data_true.title, 'UniformOutput', false);
bow_titlefact = create_bow_matrix(temp_tokens, vocabulary_title);
clear temp_tokens data_true;
disp(['✅ BoW matrix (Title Fact) created. Size: ', num2str(N_true), ' x ', num2str(length(vocabulary_title))]);

disp('Building BoW: Text Fake...');
data_fake   = readtable('Fake.csv', 'TextType', 'string');
temp_tokens = cellfun(@process_text, data_fake.text, 'UniformOutput', false);
bow_textfake = create_bow_matrix(temp_tokens, vocabulary_text);
clear temp_tokens;
disp(['✅ BoW matrix (Text Fake) created. Size: ', num2str(N_fake), ' x ', num2str(length(vocabulary_text))]);

disp('Building BoW: Title Fake...');
temp_tokens  = cellfun(@process_text, data_fake.title, 'UniformOutput', false);
bow_titlefake = create_bow_matrix(temp_tokens, vocabulary_title);
clear temp_tokens data_fake;
disp(['✅ BoW matrix (Title Fake) created. Size: ', num2str(N_fake), ' x ', num2str(length(vocabulary_title))]);

% -------------------------------------------------------------------------
% 6. FINAL WORKSPACE CLEANUP
% -------------------------------------------------------------------------
disp('--- PHASE 6: Workspace Cleanup ---');
clear i ME N_true N_fake;
disp('✅ Workspace cleaned.');

% -------------------------------------------------------------------------
% 7. SAVE RESULTS
% -------------------------------------------------------------------------
disp('--- PHASE 7: Saving Results ---');
disp('Saving BoW matrices and vocabularies to bow_results.mat...');
try
    save('bow_results.mat', ...
         'bow_textfact', 'bow_textfake', ...
         'bow_titlefact', 'bow_titlefake', ...
         'vocabulary_text', 'vocabulary_title', ...
         '-v7.3'); % -v7.3 required for files > 2GB
    disp('✅ Results saved.');
catch ME
    error('🔴 Error while saving bow_results.mat: %s', ME.message);
end
disp('-------------------------------------------------------------------');
disp('SCRIPT NUOVO.M COMPLETED.');
disp('-------------------------------------------------------------------');
