import torch
from torch import nn

torch.manual_seed(2022)

class Sequence_Modeling(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_outputs, hidden_size):
        super(Sequence_Modeling, self).__init__()
        self.emb_layer = nn.Embedding(vocab_size, embedding_size)
        self.encoder = nn.RNN(embedding_size, hidden_size, batch_first=True)
        self.mlp1 = nn.Linear(hidden_size, hidden_size)
        self.decoder = nn.RNN(embedding_size+hidden_size, hidden_size, batch_first=True)
        self.softmax = nn.Softmax(dim=2)
        self.mlp2 = nn.Linear(hidden_size, num_outputs)

    def encode(self, enc_x, state):
        enc_emb = self.emb_layer(enc_x)
        enc_hidden, state = self.encoder(enc_emb, state)

        return enc_hidden, state

    def decode(self, dec_y, enc_hidden, state, is_first_step = True):
        '''
        dec_y --> (B, S), where B = batch_size, S = sequence length
        enc_hidden --> (B, S, H), where B = batch_size, S = sequence length, H = hidden_size
        state --> (1, B, H), where B = batch_size, H = hidden_size
        is_first_step --> {True, Flase}, it is True when this is the first step of decoding, otherwise it is False
        请用RNN+attention补全解码器，其中打分函数使用点积模型
        sent_outputs的大小应为(B, S, O) where O = num_outputs, state的大小应为(1, B, H)
        '''

        dec_emb = self.emb_layer(dec_y)

        # ==========
        # todo '''请补全代码'''
        # ==========

        return sent_outputs, state


if __name__ == '__main__':
    model = Model_NP()