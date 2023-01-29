import math
from typing import Callable, List
import pandas as pd

from constants import START_OF_DOC, END_OF_DOC


# ====================
def doc_to_classes(pd_frame: pd.DataFrame) -> List:
    """
    Extract labels from dataframe
    :param pd_frame:
    :return: list of target classes/labels
    """
    return [row['label'] for (i, row) in pd_frame.iterrows()]


# ====================
def lowercase_and_strip_punctuation(word: str) -> str:
    """
    Transform a string into lowercase string without punctuation
    :param word:
    :return:
    """
    return ' '.join([c.lower() for c in word if c.isalpha() or c.isnumeric()])


# ====================
def doc_to_features(pd_frame: pd.DataFrame) -> list:
    """
    Generate a list of features for the sequence within the dataframe
    :param pd_frame: line by line sequence of tokens
    :return: the list of generated features for each token
    """
    print('Generating features')
    all_features = []

    for (i, row) in pd_frame.iterrows():
        original_word = row['token']
        word = row['clean_token']
        pos_tag = row['pos_tag']
        start_of_doc, end_of_doc = row['start_of_doc'], row['end_of_doc']
        isalnum, isupper, isnumeric, istitle = row['isalnum'], row['isupper'], row['isnumeric'], row['istitle']

        features = [
            'bias',
            'word=' + word,
            'pos_tag=' + pos_tag,
            'word.isalnum=%s' % isalnum,
            'word.isupper=%s' % isupper,
            'word.isnumeric=%s' % isnumeric,
            'word.istitle=%s' % istitle,
        ]

        if start_of_doc is True:
            features.append(START_OF_DOC)
        if end_of_doc is True:
            features.append(END_OF_DOC)

        # gets previous entries
        if i > 0:
            features.extend([
                '-1:word=' + pd_frame.iloc[i - 1]['token'],
                '-1:pos_tag=' + pd_frame.iloc[i - 1]['pos_tag']
            ])

        if i > 1:
            features.extend([
                '-2:word=' + pd_frame.iloc[i - 2]['token'],
                '-2:pos_tag=' + pd_frame.iloc[i - 2]['pos_tag']
            ])

        if i > 2:
            features.extend([
                '-3:word=' + pd_frame.iloc[i - 3]['token'],
                '-3:pos_tag=' + pd_frame.iloc[i - 3]['pos_tag']
            ])

        if i > 3:
            features.extend([
                '-4:word=' + pd_frame.iloc[i - 4]['token'],
                '-4:pos_tag=' + pd_frame.iloc[i - 4]['pos_tag']
            ])

        # get next articles
        if i < len(pd_frame) - 2:
            features.extend([
                '+1:word=' + pd_frame.iloc[i + 1]['token'],
                '+1:pos_tag=' + pd_frame.iloc[i + 2]['pos_tag']
            ])

        if i < len(pd_frame) - 3:
            features.extend([
                '+2:word=' + pd_frame.iloc[i + 2]['token'],
                '+2:pos_tag=' + pd_frame.iloc[i + 2]['pos_tag']
            ])

        if i < len(pd_frame) - 4:
            features.extend([
                '+3:word=' + pd_frame.iloc[i + 3]['token'],
                '+3:pos_tag=' + pd_frame.iloc[i + 3]['pos_tag']
            ])

        if i < len(pd_frame) - 5:
            features.extend([
                '+4:word=' + pd_frame.iloc[i + 4]['token'],
                '+4:pos_tag=' + pd_frame.iloc[i + 4]['pos_tag']
            ])

        all_features.append(features)

    return all_features
