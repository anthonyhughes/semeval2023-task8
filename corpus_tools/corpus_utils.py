import json
import re
import statistics
import string
from typing import List, Dict, Tuple

import nltk
from nltk.corpus import stopwords
import pandas as pd

from consts import TARGET_CLASSES, PICO_MAP


# nltk.download('stopwords')


def clean_token(token: str) -> str:
    """
    Clean up tokens

    :param token:
    :return:
    """
    token = token.strip()
    token = token.lower()
    token = token.replace("\n", "")
    token = token.split("http")[0]
    if len([char for char in token if char in string.punctuation]) == len(token):
        return '[UNK]'
    if token == '':
        return '[UNK]'
    if token == ' ':
        return '[UNK]'
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
    """
    Locate the text that fits within a given annotation span
    :param span:
    :param text:
    :return: the string within the span
    """
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
    if len(results) > 0:
        return results[0]
    else:
        return 'O'


def generate_corpus(word_token_spans: List, task: str = 'b') -> List:
    """
    Generate token by token corpus from the word/label spans extracted from the RedHOT corpora
    :param word_token_spans:
    :param task:
    :return: final list of tokens with annotated labels
    """
    final_corpus = []
    last_end_offset = 0
    last_start_offset = 0
    span_count = 0
    for labels, row, subreddit_id, post_id in word_token_spans:
        for token, clean_token, start, end, pos_tag, isalnum, isupper, isnumeric, istitle, has_question_mark, \
            start_of_doc, end_of_doc in \
                row:
            median_value = statistics.median([start, end])
            available_spans = get_span_for_index(median_value, labels)
            if len(available_spans) >= 1:
                start_offset, end_offset = available_spans[0]['startOffset'], available_spans[0]['endOffset']

                if last_start_offset != start_offset and last_end_offset != end_offset:
                    span_count = 0

                if span_count == 0:
                    last_start_offset, last_end_offset = start_offset, end_offset
                    prefix = 'B'

                if span_count >= 1:
                    prefix = 'I'

                span_count += 1

                final_corpus.append(
                    (
                        token,
                        clean_token,
                        lookup_class_label(available_spans[0]['label'], task, prefix),
                        pos_tag,
                        subreddit_id,
                        post_id,
                        isalnum, isupper, isnumeric, istitle, has_question_mark, start_of_doc, end_of_doc)
                )

            else:
                final_corpus.append(
                    (token,
                     clean_token,
                     'O',
                     pos_tag,
                     subreddit_id,
                     post_id,
                     isalnum, isupper, isnumeric, istitle, has_question_mark, start_of_doc, end_of_doc)
                )
    return final_corpus


def update_with_distant_annotations(corpus: List) -> List:
    stops = set(stopwords.words('english'))
    annotations = dict(set([(entry[1], entry[2]) for entry in corpus if entry[2] != 'O' and entry[1] not in stops]))
    new_corpus = []
    for entry in corpus:
        word = entry[1]
        if entry[2] == 'O':
            if word in annotations:
                pico_label = annotations[word]
                new_entry = tuple(entry[0:2] + tuple([pico_label]) + entry[3:])
                new_corpus.append(new_entry)
            else:
                new_corpus.append(entry)
        else:
            new_corpus.append(entry)
    return new_corpus


def generate_all_word_tokens_spans(all_entries_df: pd.DataFrame, target_column: str) -> List[Tuple]:
    if target_column in all_entries_df:
        word_token_spans = [
            (get_all_spans(labels), generate_word_tokens_spans(text), sub_id, post_id)
            for labels, text, sub_id, post_id in
            zip(all_entries_df[target_column],
                all_entries_df['text'],
                all_entries_df['subreddit_id'],
                all_entries_df['post_id'])]
    else:
        word_token_spans = [
            ([], generate_word_tokens_spans(text), sub_id, post_id)
            for text, sub_id, post_id in
            zip(all_entries_df['text'],
                all_entries_df['subreddit_id'],
                all_entries_df['post_id'])
        ]
    return word_token_spans


def generate_word_tokens_spans(text: str) -> List:
    """
    Generate a list of tuples containing a token, its span positions in the corpus and its pos tag
    :param text:
    :return: a list of tuples
    """
    token_spans = []
    start_of_word = True
    current_word = ''
    current_start_span = 0
    words = text.split(" ")
    text = " ".join(words)
    all_pos_tags = nltk.pos_tag(words)
    start_of_doc = True
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
                (
                    current_word,
                    cleaned_word,
                    current_start_span,
                    index - 1,
                    lookup_pos_tag(all_pos_tags, current_word),
                    current_word.isalnum(),
                    current_word.isupper(),
                    current_word.isnumeric(),
                    current_word.istitle(),
                    '?' in current_word,
                    start_of_doc,
                    True if index == len(text) - 1 else False
                )
            )
            start_of_doc = False
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


def lookup_class_label(label: str, task: str = 'b', prefix: str = 'I') -> str:
    """
    Handle PICO labels appropriately for competition submission

    :param prefix:
    :param label: to be altered
    :param task: different tasks require different labels generated
    :return:
    """

    if task == 'a' and label in TARGET_CLASSES:
        return f'{label}'
    elif task == 'a':
        return 'O'

    if task == 'b' and label in PICO_MAP:
        return PICO_MAP[label]
    else:
        return 'O'
