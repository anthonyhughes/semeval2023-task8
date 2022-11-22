import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim):
        super().__init__()

        self.embedding = nn.EmbeddingBag(input_dim, embedding_dim, sparse=True)
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        init_range = 0.5
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        # text = [sent len, batch size]
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
