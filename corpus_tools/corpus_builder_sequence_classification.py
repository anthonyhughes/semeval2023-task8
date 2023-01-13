from typing import Callable

import pandas as pd

from corpus_utils import get_all_spans, generate_span_text, preserve_label


def create_corpus_csv(file_location: str, output_location: str, is_test_set: bool, fit_label: Callable):
    """"
        Create a new CSV with 1 to 1 sentence and labels
    """
    text_for_removed_deleted_posts = r"\[deleted by user|\[removed|\[deleted"

    all_entries_df = pd.read_csv(file_location)
    all_entries_df = all_entries_df[
        all_entries_df['text'].str.contains(text_for_removed_deleted_posts) == False
        ]

    if is_test_set is not True:
        result = [
            (subid, pid, word, get_label_for_word())
            for pid, subid, labels, text in zip(all_entries_df['post_id'],
                            all_entries_df['subreddit_id'],
                            all_entries_df['stage1_labels'],
                            all_entries_df['text'])
            for index, word in enumerate_spans(word, labels)
        ]
    else:
        result = [
            (subid, pid, word, '')
            for pid, subid, text in zip(all_entries_df['post_id'],
                                                all_entries_df['subreddit_id'],
                                                all_entries_df['text'])
            for word in text.split(" ")
        ]
    final_frame = pd.DataFrame(result, columns=['subreddit_id', 'post_id', 'words', 'labels'])
    final_frame.to_csv(path_or_buf=output_location, index=False)


if __name__ == '__main__':
    create_corpus_csv(
        file_location='./medical-corpus/st1/st1_train_inc_text_.csv',
        output_location='./experiment_sequence_classification/st1_train_all_cats.csv',
        is_test_set=False,
        fit_label=preserve_label
    )
    create_corpus_csv(
        file_location='./medical-corpus/st1/st1_test_inc_text_.csv',
        output_location='./experiment_sequence_classification/st1_test_all_cats.csv',
        is_test_set=True,
        fit_label=preserve_label
    )
