import time

import torch
from torch.utils.data import DataLoader, random_split
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.data.functional import to_map_style_dataset


from datasets import RedditMedicalDataset, yield_tokens, fit_label
from experiment_simple_rnn.model import RNN
from experiment_simple_rnn.train import train, evaluate

tokenizer = get_tokenizer('basic_english')

SEED = 1234
EMBEDDING_DIM = 64
HIDDEN_DIM = 256
OUTPUT_DIM = 1
EPOCHS = 10
LR = 1
BATCH_SIZE = 64
PATH = './embedded_bag_rnn.torch'

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def text_pipeline(x, target_vocab):
    return target_vocab(tokenizer(x))


def label_pipeline(x):
    return fit_label(x)


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text, target_vocab), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


def build_vocab_from_dataset(train_set):
    new_voc = build_vocab_from_iterator(yield_tokens(iter(train_set)), specials=["<unk>"])
    new_voc.set_default_index(new_voc["<unk>"])
    return new_voc


if __name__ == '__main__':
    print('building datasets')

    dataset = RedditMedicalDataset()

    train_set, test_set = random_split(
        dataset,
        [
            len(dataset) - 1300,
            1300,
        ]
    )

    train_dataset = to_map_style_dataset(train_set)
    test_dataset = to_map_style_dataset(test_set)

    print(f'Number of train samples {len(train_set)}')
    print(f'Number of test samples {len(test_set)}')

    target_vocab = build_vocab_from_dataset(train_set)

    num_train = int(len(train_set) * 0.95)

    split_train_set, split_valid_set = \
        random_split(train_set, [num_train, len(train_set) - num_train])

    print(f'Number of split train samples {len(split_train_set)}')
    print(f'Number of split dev samples {len(split_valid_set)}')

    print('Building dataloaders')
    train_dataloader = DataLoader(split_train_set,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_set,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  collate_fn=collate_batch)
    test_dataloader = DataLoader(test_set,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True,
                                 collate_fn=collate_batch)

    print('define the model')
    vocab_size = len(target_vocab)
    print('vocab size', vocab_size)
    print('target classes', 2)
    model = RNN(input_dim=vocab_size,
                embedding_dim=EMBEDDING_DIM,
                output_dim=2
                ) \
        .to(device)

    print('start training')
    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, 1, gamma=0.1)
    total_accu = None

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(train_dataloader, model, optimiser, criterion, epoch)
        accu_val = evaluate(valid_dataloader, model, criterion)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'valid accuracy {:8.3f} '.format(epoch,
                                               time.time() - epoch_start_time,
                                               accu_val))
        print('-' * 59)

    print('Training complete')
    torch.save(model, PATH)
    print('Model weights saved')
    print('Checking the results of test dataset.')
    accu_test = evaluate(test_dataloader, model, criterion)
    print('test accuracy {:8.3f}'.format(accu_test))
