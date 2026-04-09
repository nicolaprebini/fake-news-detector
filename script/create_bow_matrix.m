function bow_matrix = create_bow_matrix(article_tokens, vocabulary)
% Crea la matrice Bag of Words (BoW) SPARTA in modo vettorializzato.
    num_articles = length(article_tokens);
    vocab_size = length(vocabulary);
    
    disp(['Inizio costruzione BoW VETTORIALIZZATA (Articoli: ', num2str(num_articles), ', Vocabolario: ', num2str(vocab_size), ')']);
    
    % Metodo Vettoriale (veloce)
    all_tokens_flat = vertcat(article_tokens{:});
    
    % Crea gli indici di riga (a quale articolo appartiene ogni token)
    article_rows_cell = cellfun(@(x,idx) repmat(idx, length(x), 1), article_tokens, num2cell(1:num_articles)', 'UniformOutput', false);
    article_rows = vertcat(article_rows_cell{:});
    
    % Trova gli indici di colonna (quale parola del vocabolario è)
    [~, final_vocab_indices] = ismember(all_tokens_flat, vocabulary);
    
    % Filtra i token che non sono nel vocabolario (indice 0)
    valid_indices = final_vocab_indices ~= 0;
    final_rows = article_rows(valid_indices);
    final_cols = final_vocab_indices(valid_indices);
    
    % accumarray costruisce la matrice sparsa contando le occorrenze
    bow_matrix = accumarray([final_rows, final_cols], 1, [num_articles, vocab_size], @sum, 0, true);
    
    disp('Costruzione BoW COMPLETATA');
end
