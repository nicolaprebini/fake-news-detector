function bow_matrix = create_bow_matrix(article_tokens, vocabulary)
% Build a sparse Bag-of-Words (BoW) matrix in a vectorised fashion.
%
% Inputs:
%   article_tokens - cell array where each cell contains the token list
%                    for one article
%   vocabulary     - cell array of unique vocabulary terms
%
% Output:
%   bow_matrix - sparse matrix of size (num_articles x vocab_size)
%                where entry (i,j) is the count of vocabulary term j
%                in article i.

    num_articles = length(article_tokens);
    vocab_size   = length(vocabulary);

    disp(['Building BoW matrix (Articles: ', num2str(num_articles), ', Vocabulary: ', num2str(vocab_size), ')']);

    % Flatten all tokens into a single array
    all_tokens_flat = vertcat(article_tokens{:});

    % Build row indices: map each token back to its source article
    article_rows_cell = cellfun(@(x,idx) repmat(idx, length(x), 1), ...
                                article_tokens, num2cell(1:num_articles)', ...
                                'UniformOutput', false);
    article_rows = vertcat(article_rows_cell{:});

    % Find column indices: position of each token in the vocabulary
    [~, final_vocab_indices] = ismember(all_tokens_flat, vocabulary);

    % Discard tokens that are not in the vocabulary (index = 0)
    valid_indices = final_vocab_indices ~= 0;
    final_rows    = article_rows(valid_indices);
    final_cols    = final_vocab_indices(valid_indices);

    % Build the sparse count matrix using accumarray
    bow_matrix = accumarray([final_rows, final_cols], 1, [num_articles, vocab_size], @sum, 0, true);

    disp('BoW matrix construction COMPLETE.');
end
