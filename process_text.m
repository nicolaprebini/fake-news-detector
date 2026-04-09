function tokens_out = process_text(text_input)
    % ===================================================================
    % PRE-PROCESSING TESTO:
    % 1. Minuscolo
    % 2. Separazione numeri/lettere
    % 3. Rimozione punteggiatura
    % 4. Normalizzazione spazi
    % 5. Split in token
    % 6. Rimozione stringhe vuote
    % 7. Rimozione stopword (inglesi + tecniche)
    % ===================================================================

    % 1. Minuscolo
    text_processed = lower(text_input); 
    
    % 2. Separa numeri e lettere (es. "2017trump" -> "2017 trump")
    text_processed = regexprep(text_processed, '(\d+)([a-z]+)', '$1 $2');
    text_processed = regexprep(text_processed, '([a-z]+)(\d+)', '$1 $2');
    
    % 3. Rimozione punteggiatura e simboli (tieni solo a-z, 0-9 e spazi)
    text_processed = regexprep(text_processed, '[^a-z0-9\s]', ' ');
    
    % 4. Spazi multipli -> singolo spazio
    text_processed = regexprep(text_processed, '\s+', ' ');
    
    % 5. Split su spazio
    tokens_out = split(text_processed, ' ');
    
    % 6. Rimuovi eventuali token vuoti
    tokens_out(strlength(tokens_out) == 0) = [];
    
    % 7. STOPWORD (inglesi + alcuni tecnici)
    stop_words = [
        "a","an","the","to","of","in","on","for","with","by","at","from", ...
        "and","or","but","if","then","else","than", ...
        "is","are","was","were","be","been","being","am","do","does","did", ...
        "it","its","this","that","these","those","there","here", ...
        "as","so","such","just","very","too","can","could","should","would", ...
        "have","has","had","will","shall","may","might","must", "reuters"...
        "mr","mrs","ms","dr", ...
        "https","http","www","com","net","org","getty","edt","re","ve","ll","s","d" ...
    ];
    
    tokens_out = tokens_out(~ismember(tokens_out, stop_words));
    
    % 8. Assicurati che l'output sia vettore colonna
    if isrow(tokens_out), tokens_out = tokens_out'; end
end
