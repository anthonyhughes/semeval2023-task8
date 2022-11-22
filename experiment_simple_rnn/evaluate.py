import torch

from datasets import RedditMedicalDataset
from experiment_simple_rnn.simple_rnn import text_pipeline, build_vocab_from_dataset
from torch.utils.data import random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels = {2: "per_exp",
          1: "question"}


def predict(text, text_pipe, target_vocab):
    with torch.no_grad():
        text = torch.tensor(text_pipe(text, target_vocab))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1


# ex_text_str = "Any quick relief remedies for the stomach cramps?"
# ex_text_str = "Do I need to stop Metoprolol prior to the TTT?"
# ex_text_str = "I once took Tylenol and hyoscyamine close together and it made me feel weird and I slept half of the day afterwards."
# ex_text_str = "The only answer ive been told is you cannot drink or eat 4 hours prior to the test"
ex_text_str = "Anyone do tube feeds at night if so did it help with gain weight?"

model = torch.load('./embedded_bag_rnn.torch')
model.eval()
model = model.to("cpu")

dataset = RedditMedicalDataset()
train_set, test_set = random_split(
    dataset,
    [
        len(dataset) - 1300,
        1300,
    ]
)
target_vocab = build_vocab_from_dataset(train_set)

print("This is a %s " % labels[predict(ex_text_str, text_pipeline, target_vocab)])
