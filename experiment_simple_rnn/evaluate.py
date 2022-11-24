import torch

from datasets import RedditMedicalDataset, transform_label
from datasets import build_vocab_from_dataset
from vocab import text_pipeline
from torch.utils.data import random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(text, text_pipe, target_vocab):
    with torch.no_grad():
        text = torch.tensor(text_pipe(text, target_vocab))
        output = model(text, torch.tensor([0]))
        result = output.argmax(1).item() + 1
        return result


ex_text_str = "n with inflammation markers at normal levels, can lupus still cause me problems/other symptoms?"
# ex_text_str = "Do I need to stop Metoprolol prior to the TTT?"
# ex_text_str = "I once took Tylenol and hyoscyamine close together and it made me feel weird and I slept half of the day afterwards."
# ex_text_str = "The only answer ive been told is you cannot drink or eat 4 hours prior to the test"
# ex_text_str = "Anyone do tube feeds at night if so did it help with gain weight?"
# ex_text_str = "Can someone explain to me what psychotic depression is?"

model = torch.load('./saved_models/embedded_bag_rnn_all_cats.pt')
model.eval()
model = model.to("cpu")

dataset = RedditMedicalDataset(target_corpus='./medical-corpus/st1_rnn/st1_train_all_cats.csv')
train_set, test_set = random_split(
    dataset,
    [
        len(dataset) - 1300,
        1300,
    ]
)
target_vocab = build_vocab_from_dataset(train_set)

print("This is a %s " % transform_label(predict(ex_text_str, text_pipeline, target_vocab)))
