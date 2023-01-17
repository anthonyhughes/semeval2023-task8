import json
import re
import string
from typing import List, Dict

import nltk
import pandas as pd


def clean_token(token: str) -> str:
    """
    Clean up tokens
    :param token:
    :return:
    """
    token = token.strip()
    token = token.lower()
    token = token.translate(str.maketrans(' ', ' ', string.punctuation))
    return token


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


def lookup_pos_tag(all_pos_tags, current_word):
    results = [pos_tag for word, pos_tag in all_pos_tags if word == current_word]
    return results[0]


def generate_word_tokens_spans(text: str) -> List:
    token_spans = []
    start_of_word = True
    current_word = ''
    current_start_span = 0
    words = re.split(r" |\n", text)
    words = [clean_token(word) for word in words]
    text = " ".join(words)
    all_pos_tags = nltk.pos_tag(words)
    for index, char in enumerate(text):
        if start_of_word is True:
            current_word = ''
            start_of_word = False
            current_word += char
            current_start_span = index
        elif char == ' ':
            start_of_word = True
            cleaned_word = clean_token(current_word)
            token_spans.append(
                (cleaned_word,
                 current_start_span,
                 index - 1,
                 lookup_pos_tag(all_pos_tags, cleaned_word))
            )
        else:
            current_word += char
    return token_spans


def get_all_spans(annotation_spans: str) -> List[Dict]:
    """
    Get all annotation labels from a row from within the original corpus
    :param annotation_spans:
    :return:
    """
    data = json.loads(annotation_spans)
    return data[0]['crowd-entity-annotation']['entities']


def get_annotation_data(dataframe: pd.DataFrame, target_row: int, target_col: str = 'stage1_labels') -> Dict:
    row = dataframe.iloc[target_row]
    return {
        'annotation_spans': row[target_col],
        'text': row['text'],
        'subreddit_id': row['subreddit_id'],
        'post_id': row['post_id'],
    }


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
