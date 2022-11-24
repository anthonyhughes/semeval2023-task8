from typing import Callable

import pandas as pd

from corpus_utils import get_all_spans, generate_span_text


def swap_label(label: str) -> str:
    """
    Force all non question labels into the per_exp category
    :param label:
    :return: new label
    """
    if label != 'question':
        return 'per_exp'
    else:
        return 'question'


def preserve_label(label: str) -> str:
    """

    :param label:
    :return:
    """
    return label


def create_corpus_csv(file_location: str, output_location: str, is_test_set: bool, fit_label: Callable):
    """"
        Create a new CSV with 1 to 1 sentence and labels
    """
    text_for_removed_deleted_posts = r"\[deleted by user|\[removed|\[deleted"
    all_entries_df = pd.read_csv(file_location)
    all_entries_df = all_entries_df.drop(columns=['post_id', 'subreddit_id'])
    all_entries_df = all_entries_df[
        all_entries_df['text'].str.contains(text_for_removed_deleted_posts) == False
    ]

    if is_test_set is not True:
        result = [
            (fit_label(span['label']), generate_span_text(span, y))
            for x, y in zip(all_entries_df['stage1_labels'], all_entries_df['text'])
            for span in get_all_spans(x)
        ]
    else:
        result = [
            ('', x)
            for x in all_entries_df['text']
        ]
    final_frame = pd.DataFrame(result, columns=['label', 'text'])
    final_frame.to_csv(path_or_buf=output_location, index=False)


if __name__ == '__main__':
    create_corpus_csv(
        file_location='./medical-corpus/st1/st1_train_inc_text_.csv',
        output_location='./medical-corpus/st1_rnn/st1_train_all_cats.csv',
        is_test_set=False,
        fit_label=preserve_label
    )
    create_corpus_csv(
        file_location='./medical-corpus/st1/st1_test_inc_text_.csv',
        output_location='./medical-corpus/st1_rnn/st1_test_all_cats.csv',
        is_test_set=True,
        fit_label=preserve_label
    )
