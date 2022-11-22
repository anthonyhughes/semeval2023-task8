from torch.utils.data import Dataset
import pandas as pd
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer('basic_english')


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


def fit_label(label: str) -> int:
    if label == 'question':
        return 0
    else:
        return 1


class RedditMedicalDataset(Dataset):
    def __init__(self, target_corpus: str = './medical-corpus/st1_rnn/st1_train.csv'):
        self.samples = pd.read_csv(target_corpus)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples.iloc[idx]['text']
        label = self.samples.iloc[idx]['label']
        return label, text


if __name__ == '__main__':
    dataset = RedditMedicalDataset()
    train_iter = iter(dataset)
    print(next(train_iter))
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

