import argparse
from typing import Dict, List

import pandas as pd

from corpus_utils import get_annotation_data, get_all_spans


def matching_class(target_clazz: str, span_class: str):
    return target_clazz == span_class


def build_stats(annotations: List) -> Dict:
    target_class_stats = {
        'claim': 0,
        'per_exp': 0,
        'claim_per_exp': 0,
        'question': 0
    }

    for key in list(target_class_stats.keys()):
        current_value = target_class_stats[key]
        length = len(list(filter(lambda x: matching_class(x['label'], key), annotations)))
        target_class_stats[key] = current_value + length
    return target_class_stats


def fetch_corpus_stats(file_location: str):
    print('Looking for file - ', file_location)
    all_entries_df = pd.read_csv(file_location)
    length = len(all_entries_df)
    print('Row count of file - ', length)
    print('Head of file - ', all_entries_df.head)
    print('Columns of file - ', all_entries_df.columns)

    all_annotations = []
    for i in range(0, length):
        corpus_entry = get_annotation_data(dataframe=all_entries_df, target_row=i)
        corpus_entry_annotations = get_all_spans(corpus_entry['annotation_spans'])
        all_annotations.extend(corpus_entry_annotations)

    stats = build_stats(all_annotations)
    print('Class - ', stats)

    # # print('Annotations in row', for_viewing['annotation_spans'])
    # # print('All spans in text', all_spans_for_viewing)
    # # print('Text in row', for_viewing['text'])
    # visualise_text_with_spans(post_id=for_viewing['post_id'],
    #                           category=for_viewing['subreddit_id'],
    #                           text=for_viewing['text'],
    #                           spans=all_spans_for_viewing)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="SemEval 2023 (Task 8) - Get basic corpus stats"
    )
    parser.add_argument("--file_location",
                        type=str,
                        help="CSV location with List of reddit post and their annotations",
                        default='./medical-corpus/st1/st1_train_inc_text_.csv'
                        )
    args = parser.parse_args()

    fetch_corpus_stats(file_location=args.file_location)
