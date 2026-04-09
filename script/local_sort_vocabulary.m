function sorted_vocab = local_sort_vocabulary(token_array)
    % Sort vocabulary: alphabetical tokens first (case-insensitive),
    % then mixed/symbol tokens, then purely numeric tokens last.

    % 1. Flatten all tokens and keep unique entries
    all_tokens_flat = vertcat(token_array{:});
    unique_tokens   = unique(all_tokens_flat, 'stable');

    % 2. Classify each token

    % A) Purely numeric tokens
    is_numeric = cellfun(@(s) all(isstrprop(s, 'digit')) && ~isempty(s), unique_tokens);

    % B) Purely alphabetic tokens
    is_alpha_only = cellfun(@(s) all(isstrprop(s, 'alpha')) && ~isempty(s), unique_tokens);

    % C) Mixed / symbol tokens (everything else)
    is_symbol_or_mixed = ~(is_numeric | is_alpha_only);

    % 3. Separate into groups

    % Group 1: alphabetic tokens — sorted case-insensitively
    alpha_tokens       = unique_tokens(is_alpha_only);
    alpha_tokens_lower = lower(alpha_tokens);
    [~, sort_indices]  = sortrows(alpha_tokens_lower);
    alpha_tokens       = alpha_tokens(sort_indices);

    % Group 2: mixed / symbol tokens
    symbol_tokens = unique_tokens(is_symbol_or_mixed);

    % Group 3: purely numeric tokens
    numeric_tokens = unique_tokens(is_numeric);

    % 4. Concatenate groups
    sorted_vocab = [alpha_tokens; symbol_tokens; numeric_tokens];
end
