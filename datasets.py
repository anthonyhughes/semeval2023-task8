from torch.utils.data import Dataset
import pandas as pd
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import random_split
from torchtext.data.utils import get_tokenizer


def yield_tokens(data_iter, tokenizer=get_tokenizer('basic_english')):
    for _, text in data_iter:
        yield tokenizer(text)


def fit_label(label: str) -> int:
    if label == 'question':
        return 1
    elif label == 'claim':
        return 2
    elif label == 'per_exp':
        return 3
    elif label == 'claim_per_exp':
        return 4


def transform_label(label: int) -> str:
    if label == 1:
        return 'question'
    elif label == 2:
        return 'claim'
    elif label == 3:
        return 'per_exp'
    elif label == 4:
        return 'claim_per_exp'


class RedditMedicalDataset(Dataset):
    def __init__(self, target_corpus: str = './medical-corpus/st1_rnn/st1_train.csv'):
        self.samples = pd.read_csv(target_corpus)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples.iloc[idx]['text']
        label = self.samples.iloc[idx]['label']
        return label, text


def generate_all_datasets(target_file: str = './medical-corpus/st1_rnn/st1_train.csv'):
    reddit_dataset = RedditMedicalDataset(target_corpus=target_file)

    train_set, test_set = random_split(
        reddit_dataset,
        [
            len(reddit_dataset) - 1300,
            1300,
        ]
    )

    print(f'Number of train samples {len(train_set)}')
    print(f'Number of test samples {len(test_set)}')

    target_vocab = build_vocab_from_dataset(train_set)

    num_train = int(len(train_set) * 0.95)

    split_train_set, split_valid_set = \
        random_split(
            train_set,
            [num_train, len(train_set) - num_train]
        )

    print(f'Number of split train samples {len(split_train_set)}')
    print(f'Number of split dev samples {len(split_valid_set)}')

    return split_train_set, split_valid_set, test_set, target_vocab


def build_vocab_from_dataset(train_set):
    new_voc = build_vocab_from_iterator(yield_tokens(iter(train_set)), specials=["<unk>"])
    new_voc.set_default_index(new_voc["<unk>"])
    return new_voc


if __name__ == '__main__':
    pass
    # dataset = RedditMedicalDataset()
    # train_iter = iter(dataset)
    # print(next(train_iter))
    # vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    # vocab.set_default_index(vocab["<unk>"])
