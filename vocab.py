import pandas as pd
import unicodedata
import string
import torch

all_letters = string.ascii_letters + " .,;'?"
n_letters = len(all_letters)


def unicode_to_ascii(s) -> str:
    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    if s is None:
        return ''
    try:
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in all_letters
        )
    except Exception as e:
        # print(e)
        # print(s)
        return ''


def load_vocabs(target_corpus: str = './medical-corpus/st1_rnn/st1_train.csv'):
    # Build the label and the associated text dictionary
    # e.g a list of names per question or expression
    category_lines = {
        'per_exp': [],
        'question': [],
    }
    all_categories = set()

    frame = pd.read_csv(target_corpus)
    for idx, sample in frame.iterrows():
        category = sample['label']
        all_categories.add(category)
        line = unicode_to_ascii(sample['text'])
        category_lines[category].append(line)

    n_categories = len(all_categories)

    print('Categories!', n_categories)

    return category_lines, all_categories, all_letters


def letter_to_index(letter):
    # Find letter index from all_letters, e.g. "a" = 0
    return all_letters.find(letter)


def letter_to_tensor(letter):
    # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


def line_to_tensor(line):
    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor


if __name__ == '__main__':
    print('Testing unicode!')
    print('Unicode example', unicode_to_ascii('Ślusàrski'))
    print('Question example', load_vocabs()[0]['question'][0])
    print('J to tensor', letter_to_tensor('J'))
    print('Jones', line_to_tensor('Jones').size())
