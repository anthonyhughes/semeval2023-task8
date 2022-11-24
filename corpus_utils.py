import json
from typing import List, Dict
import pandas as pd


def index_is_within_offsets(index: int, start_offset: int, end_offset: int) -> bool:
    return start_offset <= index < end_offset


def get_span_for_index(index: int, spans: List) -> List:
    results = filter(lambda x: index_is_within_offsets(index, x['startOffset'], x['endOffset']), spans)
    return list(results)


def has_matching_span(index: int, spans: List) -> bool:
    results = filter(lambda x: x['startOffset'] <= index < x['endOffset'], spans)
    return len(list(results)) == 1


def generate_span_text(span, text) -> str:
    filtered_annotated_text = ''
    for index, char in enumerate(text):
        if index_is_within_offsets(index, start_offset=span['startOffset'], end_offset=span['endOffset']):
            filtered_annotated_text += char
    if len(filtered_annotated_text) == 0:
        print(span)
        print(text)
    return filtered_annotated_text


def get_all_spans(annotation_spans: str) -> List[Dict]:
    """
    Get all annotation labels from a row from within the original corpus
    :param annotation_spans:
    :return:
    """
    data = json.loads(annotation_spans)
    return data[0]['crowd-entity-annotation']['entities']


def get_annotation_data(dataframe: pd.DataFrame, target_row: int) -> Dict:
    row = dataframe.iloc[target_row]
    return {
        'annotation_spans': row['stage1_labels'],
        'text': row['text'],
        'subreddit_id': row['subreddit_id'],
        'post_id': row['post_id'],
    }
