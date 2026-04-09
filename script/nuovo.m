clear; clc; 
% -------------------------------------------------------------------------
% 1. CARICAMENTO DATI (Per FASE 2-4: Costruzione Vocabolari)
% -------------------------------------------------------------------------
disp('--- FASE 1: Caricamento Dati ---');
try
    data_true = readtable('True.csv', 'TextType', 'string'); 
    data_fake = readtable('Fake.csv', 'TextType', 'string');
    disp('✅ Dati caricati con successo da True.csv e Fake.csv.');
catch ME
    if strcmp(ME.identifier, 'MATLAB:readtable:FileOpenError')
        error('🔴 Errore: Assicurati che i file True.csv e Fake.csv siano nella cartella corrente di MATLAB.');
    else
        rethrow(ME);
    end
end
% -------------------------------------------------------------------------
% 2. PRE-ALLOCAZIONE E TOKENIZZAZIONE CON CONTROLLI DI SICUREZZA
% -------------------------------------------------------------------------
disp('--- FASE 2: Tokenizzazione Completa ---');
N_true = height(data_true);
N_fake = height(data_fake);
textfact = cell(N_true, 1); 
textfake = cell(N_fake, 1); 
titlefact = cell(N_true, 1); 
titlefake = cell(N_fake, 1); 
disp('Inizio elaborazione: tokenizzazione True/Fact...');
for i = 1:N_true
    if mod(i, 1000) == 0
        disp(['  > Tokenizzazione True Progress: Articolo ', num2str(i), ' di ', num2str(N_true)]);
    end
    current_text = data_true.text(i);
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
disp('Inizio elaborazione: tokenizzazione Fake...');
for i = 1:N_fake
    if mod(i, 1000) == 0
        disp(['  > Tokenizzazione Fake Progress: Articolo ', num2str(i), ' di ', num2str(N_fake)]);
    end
    current_text = data_fake.text(i);
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
disp('✅ Tokenizzazione completata.');
% -------------------------------------------------------------------------
% 3. CREAZIONE E RIORDINAMENTO DEI VOCABOLARI
% -------------------------------------------------------------------------
disp('--- FASE 3: Creazione e Riordinamento Vocabolari ---');
all_text_tokens = [textfact; textfake];
vocabulary_text = local_sort_vocabulary(all_text_tokens);
disp(['✅ Vocabolario del Testo creato e ordinato. Totale iniziale: ', num2str(length(vocabulary_text))]);
all_title_tokens = [titlefact; titlefake];
vocabulary_title = local_sort_vocabulary(all_title_tokens);
disp(['✅ Vocabolario dei Titoli creato e ordinato. Totale iniziale: ', num2str(length(vocabulary_title))]);
% -------------------------------------------------------------------------
% 4. FILTRAGGIO DEI VOCABOLARI (RIMUOVI OCCORRENZE SINGOLE)
% -------------------------------------------------------------------------
disp('--- FASE 4: Filtraggio Vocabolari (Rimozione token unici) ---');
[vocabulary_text_filtered, vocabulary_title_filtered] = filter_vocabulary_by_frequency(...
    textfact, textfake, titlefact, titlefake, ... 
    vocabulary_text, vocabulary_title);          
vocabulary_text = vocabulary_text_filtered;
vocabulary_title = vocabulary_title_filtered;
disp('✅ Token unici (frequenza = 1) rimossi dai vocabolari.');
disp(['Nuovo totale token Testo: ', num2str(length(vocabulary_text))]);
disp(['Nuovo totale token Titoli: ', num2str(length(vocabulary_title))]);

% 💡 --- INIZIO OTTIMIZZAZIONE RAM ---
disp('Pulizia variabili token intermedie per liberare RAM...');
clear textfact textfake titlefact titlefake ...
      all_text_tokens all_title_tokens ...
      vocabulary_text_filtered vocabulary_title_filtered ...
      data_true data_fake current_text current_title;
% 💡 --- FINE OTTIMIZZAZIONE RAM ---

% -------------------------------------------------------------------------
% 5. CREAZIONE DELLE MATRICI BAG OF WORDS (COMPLETA E OTTIMIZZATA)
% -------------------------------------------------------------------------
disp('--- FASE 5: Creazione Matrici BoW (COMPLETA E OTTIMIZZATA) ---');
disp('Creazione BoW: Text Fact...');
data_true = readtable('True.csv', 'TextType', 'string'); 
temp_tokens = cellfun(@process_text, data_true.text, 'UniformOutput', false);
bow_textfact = create_bow_matrix(temp_tokens, vocabulary_text); 
clear temp_tokens; 
disp(['✅ Matrice BoW (Text Fact) creata. Dimensione: ', num2str(N_true), ' x ', num2str(length(vocabulary_text))]);

disp('Creazione BoW: Title Fact...');
temp_tokens = cellfun(@process_text, data_true.title, 'UniformOutput', false);
bow_titlefact = create_bow_matrix(temp_tokens, vocabulary_title);
clear temp_tokens data_true; 
disp(['✅ Matrice BoW (Title Fact) creata. Dimensione: ', num2str(N_true), ' x ', num2str(length(vocabulary_title))]);

disp('Creazione BoW: Text Fake...');
data_fake = readtable('Fake.csv', 'TextType', 'string');
temp_tokens = cellfun(@process_text, data_fake.text, 'UniformOutput', false);
bow_textfake = create_bow_matrix(temp_tokens, vocabulary_text);
clear temp_tokens; 
disp(['✅ Matrice BoW (Text Fake) creata. Dimensione: ', num2str(N_fake), ' x ', num2str(length(vocabulary_text))]);

disp('Creazione BoW: Title Fake...');
temp_tokens = cellfun(@process_text, data_fake.title, 'UniformOutput', false);
bow_titlefake = create_bow_matrix(temp_tokens, vocabulary_title);
clear temp_tokens data_fake; 
disp(['✅ Matrice BoW (Title Fake) creata. Dimensione: ', num2str(N_fake), ' x ', num2str(length(vocabulary_title))]);

% -------------------------------------------------------------------------
% 6. PULIZIA FINALE DEL WORKSPACE
% -------------------------------------------------------------------------
disp('--- FASE 6: Pulizia Workspace ---');
clear i ME N_true N_fake; 
disp('✅ Workspace pulito.');

% -------------------------------------------------------------------------
% 7. SALVATAGGIO RISULTATI
% -------------------------------------------------------------------------
disp('--- FASE 7: Salvataggio Risultati ---');
disp('Salvataggio di matrici BoW e vocabolari in bow_results.mat...');
try
    save('bow_results.mat', ...
         'bow_textfact', 'bow_textfake', ...
         'bow_titlefact', 'bow_titlefake', ...
         'vocabulary_text', 'vocabulary_title', ...
         '-v7.3'); % -v7.3 è per file > 2GB
    disp('✅ Risultati salvati.');
catch ME
    error('🔴 Errore durante il salvataggio di bow_results.mat: %s', ME.message);
end
disp('-------------------------------------------------------------------');
disp('SCRIPT NUOVO.M COMPLETATO.');
disp('-------------------------------------------------------------------');
