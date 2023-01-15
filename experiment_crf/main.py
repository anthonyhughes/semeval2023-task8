import statistics
import string

from corpus_tools.corpus_utils import get_all_spans, generate_word_tokens_spans, \
    get_span_for_index
import pandas as pd


def clean_token(token: str) -> str:
    """
    Clean up tokens
    :param token:
    :return:
    """
    token = token.strip()
    token = token.lower()
    token = token.translate(str.maketrans('', '', string.punctuation))
    return token


def is_bad_token(token: str) -> bool:
    return token == "" or token.startswith("http")


def create_token_classification_corpus_as_csv(file_location: str,
                                              output_location: str,
                                              target_column: str = 'stage1_labels') -> None:
    """"
        Create a new CSV with 1 to 1 sentence and labels
    """
    text_for_removed_deleted_posts = r"\[deleted by user|\[removed|\[deleted"
    all_entries_df = pd.read_csv(file_location)
    all_entries_df = all_entries_df.drop(columns=['post_id', 'subreddit_id'])
    all_entries_df = all_entries_df[
        all_entries_df['text'].str.contains(text_for_removed_deleted_posts) == False
        ]

    if target_column in all_entries_df:
        word_token_spans = [(get_all_spans(labels), generate_word_tokens_spans(text)) for labels, text in
                            zip(all_entries_df[target_column],
                                all_entries_df['text'])]
    else:
        word_token_spans = [([], generate_word_tokens_spans(text)) for text in all_entries_df['text']]
    final_corpus = []
    for labels, row in word_token_spans:
        for token, start, end in row:
            median_value = statistics.median([start, end])
            available_spans = get_span_for_index(median_value, labels)
            token = clean_token(token)
            if is_bad_token(token):
                continue
            if len(available_spans) >= 1:
                final_corpus.append((token, available_spans[0]['label']))
            else:
                final_corpus.append((token, 'none'))
    final_frame = pd.DataFrame(final_corpus, columns=['token', 'label'])
    final_frame.to_csv(path_or_buf=output_location, index=False)


if __name__ == '__main__':
    create_token_classification_corpus_as_csv(
        file_location='./medical-corpus/st2/st2_train_inc_text_.csv',
        output_location='./experiment_crf/st2_train_all_cats.csv',
        target_column='stage2_labels'
    )
    create_token_classification_corpus_as_csv(
        file_location='./medical-corpus/st2/st2_test_inc_text_.csv',
        output_location='./experiment_crf/st2_test_all_cats.csv',
        target_column='stage2_labels'
    )
