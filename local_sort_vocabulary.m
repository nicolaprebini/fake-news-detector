function sorted_vocab = local_sort_vocabulary(token_array)
    % Ordina il vocabolario: Alfabetico (case-insensitive) prima, Simboli, Numeri alla fine.
    
    % 1. Appiattimento e Unicità
    all_tokens_flat = vertcat(token_array{:});
    unique_tokens = unique(all_tokens_flat, 'stable'); 
    
    % 2. Identificazione dei token
    
    % A) Token che contengono SOLO numeri.
    is_numeric = cellfun(@(s) all(isstrprop(s, 'digit')) && ~isempty(s), unique_tokens);
    
    % B) Token che contengono SOLO lettere (alpha)
    is_alpha_only = cellfun(@(s) all(isstrprop(s, 'alpha')) && ~isempty(s), unique_tokens);
    
    % C) Simboli/Punteggiatura e token misti sono il resto.
    is_symbol_or_mixed = ~(is_numeric | is_alpha_only);
    
    % 3. Separazione dei gruppi
    
    % Gruppo 1: Solo Parole
    alpha_tokens = unique_tokens(is_alpha_only);
    
    % 💡 CORREZIONE DI COMPATIBILITÀ: Ordinamento Case-Insensitive Manuale
    alpha_tokens_lower = lower(alpha_tokens); 
    [~, sort_indices] = sortrows(alpha_tokens_lower); 
    alpha_tokens = alpha_tokens(sort_indices); 
    
    % Gruppo 2: Simboli/Punteggiatura e Token Misti 
    symbol_tokens = unique_tokens(is_symbol_or_mixed);
    
    % Gruppo 3: Solo Numeri
    numeric_tokens = unique_tokens(is_numeric); 
    
    % 4. Ri-combinazione
    sorted_vocab = [alpha_tokens; symbol_tokens; numeric_tokens];
end