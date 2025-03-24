import torch
from torch import nn


class Sequence_Modeling(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_outputs, hidden_size):
        super(Sequence_Modeling, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.step = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_outputs)

    def forward(self, sent, state):
        sent_emb = self.embeddings(sent)

        steps = sent.size(1)
        sent_hidden, state = self.step(sent_emb, state)
        sent_states = self.linear(sent_hidden)

        return sent_states, state


if __name__ == '__main__':
    model = Model_NP()