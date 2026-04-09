function [new_vocab_text, new_vocab_title] = filter_vocabulary_by_frequency(text_fact, text_fake, title_fact, title_fake, vocab_text, vocab_title)
% Filtra i vocabolari rimuovendo tutte le occorrenze che appaiono solo una volta
% nell'intero corpus di testo o titoli combinati.

    % =========================================================
    % 1. ANALISI FREQUENZA DEL TESTO
    % =========================================================
    all_text_tokens_flat = vertcat(text_fact{:}, text_fake{:});
    [unique_tokens_text, ~, token_indices] = unique(all_text_tokens_flat);
    token_counts_text = histcounts(token_indices, 1:length(unique_tokens_text)+1);
    tokens_to_keep_text = unique_tokens_text(token_counts_text > 1);
    
    % =========================================================
    % 2. ANALISI FREQUENZA DEI TITOLI
    % =========================================================
    all_title_tokens_flat = vertcat(title_fact{:}, title_fake{:});
    [unique_tokens_title, ~, token_indices_title] = unique(all_title_tokens_flat);
    token_counts_title = histcounts(token_indices_title, 1:length(unique_tokens_title)+1);
    tokens_to_keep_title = unique_tokens_title(token_counts_title > 1);
    
    % =========================================================
    % 3. FILTRAGGIO E AGGIORNAMENTO DEI VOCABOLARI
    % =========================================================
    new_vocab_text = vocab_text(ismember(vocab_text, tokens_to_keep_text));
    new_vocab_title = vocab_title(ismember(vocab_title, tokens_to_keep_title));
end