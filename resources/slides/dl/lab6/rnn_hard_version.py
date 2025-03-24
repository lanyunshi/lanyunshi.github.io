import torch
from torch import nn
import numpy as np

np.random.seed(2022)
torch.manual_seed(2022)

class GRU(nn.Module):
    def __init__(self, num_inputs, num_hiddens):
        super(GRU, self).__init__()
        # 隐藏层参数
        '''
        请声明GRU中的各类参数
        '''


    def forward(self, inputs, H):
        '''
        利用定义好的参数补全GRU的前向传播，
        不能调用pytorch中内置的GRU函数及操作
        '''
        # ==========
        # todo '''请补全GRU网络前向传播'''
        # ==========
        return outputs, H


class Sequence_Modeling(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_outputs, hidden_size):
        super(Sequence_Modeling, self).__init__()
        self.emb_layer = nn.Embedding(vocab_size, embedding_size)
        self.gru_layer = GRU(embedding_size, hidden_size)
        self.linear = nn.Linear(hidden_size, num_outputs)

    def forward(self, sent, state):
        '''
        sent --> (B, S) where B = batch size, S = sequence length
        sent_emb --> (B, S, I) where B = batch size, S = sequence length, I = num_inputs
        state --> (B, 1, H), where B = batch_size, num_hiddens
        你需要利用定义好的emb_layer, gru_layer和linear，
        补全代码实现歌词预测功能，
        sent_outputs的大小应为(B, S, O) where O = num_outputs, state的大小应为(B, 1, H)
        '''

        sent_emb = self.embeddings(sent)

        # ==========
        # todo '''请补全代码'''
        # ==========
        return sent_outputs, state


if __name__ == '__main__':
    model = Sequence_Modeling()