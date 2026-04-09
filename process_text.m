function tokens_out = process_text(text_input)
    % ===================================================================
    % TEXT PRE-PROCESSING PIPELINE:
    % 1. Lowercase conversion
    % 2. Number/letter separation
    % 3. Punctuation removal
    % 4. Whitespace normalization
    % 5. Tokenization (split into tokens)
    % 6. Empty token removal
    % 7. Stop-word removal (English + technical)
    % ===================================================================

    % 1. Lowercase
    text_processed = lower(text_input);

    % 2. Separate numbers and letters (e.g. "2017trump" -> "2017 trump")
    text_processed = regexprep(text_processed, '(\d+)([a-z]+)', '$1 $2');
    text_processed = regexprep(text_processed, '([a-z]+)(\d+)', '$1 $2');

    % 3. Remove punctuation and symbols (keep only a-z, 0-9 and spaces)
    text_processed = regexprep(text_processed, '[^a-z0-9\s]', ' ');

    % 4. Collapse multiple spaces into a single space
    text_processed = regexprep(text_processed, '\s+', ' ');

    % 5. Split on whitespace
    tokens_out = split(text_processed, ' ');

    % 6. Remove empty tokens
    tokens_out(strlength(tokens_out) == 0) = [];

    % 7. Stop-word list (English function words + technical tokens)
    stop_words = [
        "a","an","the","to","of","in","on","for","with","by","at","from", ...
        "and","or","but","if","then","else","than", ...
        "is","are","was","were","be","been","being","am","do","does","did", ...
        "it","its","this","that","these","those","there","here", ...
        "as","so","such","just","very","too","can","could","should","would", ...
        "have","has","had","will","shall","may","might","must","reuters", ...
        "mr","mrs","ms","dr", ...
        "https","http","www","com","net","org","getty","edt","re","ve","ll","s","d" ...
    ];

    tokens_out = tokens_out(~ismember(tokens_out, stop_words));

    % 8. Ensure output is a column vector
    if isrow(tokens_out), tokens_out = tokens_out'; end
end
