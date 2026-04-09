clc;
clear;

% -------------------------------------------------------------------------
% 1. CARICAMENTO MODELLI, VOCABOLARI E IDF
% -------------------------------------------------------------------------
disp('--- FASE 1: Caricamento Modelli, Vocabolari e IDF ---');
try
    load('bow_results.mat', 'Mdl_Title', 'Mdl_Text', ...
         'vocabulary_title', 'vocabulary_text', ...
         'idf_title', 'idf_text');
    disp('✅ Modelli (Mdl_Title, Mdl_Text), vocabolari e IDF caricati.');
catch
    error(['🔴 Errore: impossibile caricare Mdl_Title, Mdl_Text, ' ...
           'vocabulary_*, idf_* da bow_results.mat.']);
end

% -------------------------------------------------------------------------
% 2. CARICAMENTO E FILTRAGGIO DATI
% -------------------------------------------------------------------------
disp('--- FASE 2: Caricamento Dati (url-versions-2015-06-14.csv) ---');

fileName = 'url-versions-2015-06-14.csv';

try
    data = readtable(fileName, 'TextType', 'string');
    disp(['✅ File "', fileName, '" caricato correttamente.']);
catch
    error(['🔴 Errore: Impossibile leggere il file ', fileName, ...
           '. Verifica che sia nella stessa cartella dello script.']);
end

% Controllo colonne comuni
requiredVars = ["claimHeadline", "claimTruthiness"];
if ~all(ismember(requiredVars, string(data.Properties.VariableNames)))
    error('🔴 Il file deve contenere "claimHeadline" e "claimTruthiness".');
end

% Colonna TESTO (modifica se necessario)
textColumnName = "articleBody";   % <--- CAMBIA QUI se ha un altro nome
if ~ismember(textColumnName, string(data.Properties.VariableNames))
    error(['🔴 Colonna di testo "', textColumnName, '" non trovata nel CSV. ', ...
           'Modifica textColumnName nello script.']);
end

% Filtra solo true/false
disp('Filtraggio righe con claimTruthiness = "true" o "false"...');
truthiness = data.claimTruthiness;
mask_valid = (truthiness == "true") | (truthiness == "false");
data_valid = data(mask_valid, :);

disp(['Totale righe nel file: ', num2str(height(data))]);
disp(['Righe usate per il test (true/false): ', num2str(height(data_valid))]);

if height(data_valid) == 0
    error('🔴 Nessuna riga con claimTruthiness = "true" o "false". Impossibile procedere.');
end

% Etichette comuni
labels_raw = data_valid.claimTruthiness;      % "true"/"false"
Y_test = strings(height(data_valid), 1);
Y_test(labels_raw == "true")  = "Fact";
Y_test(labels_raw == "false") = "Fake";
Y_test_str = string(Y_test);

% -------------------------------------------------------------------------
% 3. TEST SUI TITOLI (Mdl_Title, vocabulary_title, idf_title)
% -------------------------------------------------------------------------
disp(' ');
disp('===================================================================');
disp('TEST 1: TITOLI (claimHeadline) con Mdl_Title');
disp('===================================================================');

titles = data_valid.claimHeadline;

disp('Tokenizzazione dei TITOLI...');
tokens_title = cellfun(@process_text, titles, 'UniformOutput', false);

disp('Creazione BoW per i TITOLI...');
bow_test_title = create_bow_matrix(tokens_title, vocabulary_title);

% Applica TF-IDF con idf_title del training
X_test_title = bow_test_title .* idf_title;

disp('Predizione con Mdl_Title...');
Y_pred_title = predict(Mdl_Title, X_test_title);
Y_pred_title_str = string(Y_pred_title);

accuracy_title = mean(Y_pred_title_str == Y_test_str);

disp('-------------------------------------------------------------------');
disp(['✅ ACCURATEZZA sui TITOLI: ', num2str(accuracy_title * 100, '%.2f'), '%']);
disp('-------------------------------------------------------------------');



% Matrice di confusione TITOLI
figure;
confusionchart(Y_test_str, Y_pred_title_str);
title('Matrice di Confusione - TITOLI (url-versions-2015-06-14)');


% -------------------------------------------------------------------------
% 4. TEST SUI TESTI (Mdl_Text, vocabulary_text, idf_text)
% -------------------------------------------------------------------------
disp(' ');
disp('===================================================================');
disp(['TEST 2: TESTI (', char(textColumnName), ') con Mdl_Text']);
disp('===================================================================');

texts = data_valid.(textColumnName);

disp('Tokenizzazione dei TESTI...');
tokens_text = cellfun(@process_text, texts, 'UniformOutput', false);

disp('Creazione BoW per i TESTI...');
bow_test_text = create_bow_matrix(tokens_text, vocabulary_text);

% Applica TF-IDF con idf_text del training
X_test_text = bow_test_text .* idf_text;

disp('Predizione con Mdl_Text...');
Y_pred_text = predict(Mdl_Text, X_test_text);
Y_pred_text_str = string(Y_pred_text);

accuracy_text = mean(Y_pred_text_str == Y_test_str);

disp('-------------------------------------------------------------------');
disp(['✅ ACCURATEZZA sui TESTI: ', num2str(accuracy_text * 100, '%.2f'), '%']);
disp('-------------------------------------------------------------------');

% Matrice di confusione TESTI
figure;
confusionchart(Y_test_str, Y_pred_text_str);
title(['Matrice di Confusione - TESTI (', char(textColumnName), ', url-versions-2015-06-14)']);

disp('===================================================================');
disp('FINE SCRIPT test_urlversions.m (Titoli + Testi).');
disp('===================================================================');
