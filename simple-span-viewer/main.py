import argparse
from typing import List
from colorama import Style

from consts import COLOUR_MAPPING, SUBREDDIT_ID_TO_POPULATION
import pandas as pd

from corpus_tools.corpus_utils import get_annotation_data, get_all_spans, get_span_for_index


def visualise_text_with_spans(post_id: str, category: str, text: str, spans: List) -> None:
    print('#########')
    print('Starting Visualisation \n')
    print(f'Post ID - {post_id}')
    print(f'Subreddit label - {SUBREDDIT_ID_TO_POPULATION[category]}')
    print(f'Spans count - {len(spans)}')
    for index, char in enumerate(text):
        filtered_spans = get_span_for_index(index=index, spans=spans)

        if len(filtered_spans) == 1:
            print(COLOUR_MAPPING[filtered_spans[0]['label']], char, sep='', end='')
        else:
            print(char, sep='', end='')

        print(Style.RESET_ALL, end='')
    print('\n#########')


def run_viewer(file_location: str, target_row: int) -> None:
    """
    Main function for running the annotation viewer
    :param target_row:
    :param file_location: location of text and annotations
    :return:
    """
    all_entries_df = pd.read_csv(file_location)
    # print('Example row', all_entries_df.iloc[0])
    for_viewing = get_annotation_data(dataframe=all_entries_df, target_row=target_row)
    # print('Annotations in row', for_viewing['annotation_spans'])
    all_spans_for_viewing = get_all_spans(for_viewing['annotation_spans'])
    # print('All spans in text', all_spans_for_viewing)
    # print('Text in row', for_viewing['text'])
    visualise_text_with_spans(post_id=for_viewing['post_id'],
                              category=for_viewing['subreddit_id'],
                              text=for_viewing['text'],
                              spans=all_spans_for_viewing)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="SemEval 2023 (Task 8) - Parsing arguments for viewing annotations of a reddit post"
    )
    parser.add_argument("--file_location",
                        type=str,
                        help="CSV location with List of reddit post and their annotations",
                        required=True
                        )
    parser.add_argument("--target_row",
                        type=int,
                        help="Row in CSV to be visualised",
                        required=True
                        )
    args = parser.parse_args()

    run_viewer(file_location=args.file_location, target_row=args.target_row)
