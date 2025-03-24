import torch
from torch import nn
import math
from rnn_hard_version_solution import Sequence_Modeling
from rnn_easy_version import Sequence_Modeling as Sequence_Modeling_pytorch

torch.manual_seed(2022)
PUNCTUATION = '''!"#$%&'()*+, -./:;<=>?@[\]^_`{|}~'''

def load_jaychou_lyrics():
    with open('data/jaychou_train.txt') as f: #, errors='ignore'
        train_chars = f.readlines()
    train_chars = (''.join(train_chars)).replace('\n', ' ').replace('\r', ' ')

    with open('data/jaychou_test_X.txt') as f:
        test_chars = f.readlines()
    test_chars = [sent.replace('\n', '') for sent in test_chars]

    return train_chars, test_chars

def text_processing(train_chars, test_chars, num_steps):
    word2idx, idx2word = {'PAD': 0, 'UNK': 1}, {0: 'PAD', 1: 'UNK'}
    train_ids, train_labels, test_ids, test_labels = [], [], [], []

    for w in train_chars:
        if w not in word2idx:
            word2idx[w] = len(word2idx)
            idx2word[word2idx[w]] = w
        train_id = word2idx[w] if w in word2idx else word2idx['UNK']
        train_ids += [train_id]

    for sent in test_chars:
        if len(sent) != num_steps: print(sent); print(len(sent)); exit()
        test_id = [word2idx[w] if w in word2idx else word2idx['UNK'] for w in sent]
        test_ids += [test_id]

    return train_ids, test_ids, word2idx, idx2word

def data_iter_consecutive(corpus_indices, batch_size, num_steps):
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield torch.tensor(X, dtype=torch.int), torch.tensor(Y, dtype=torch.int)

def predict_rnn_pytorch(prefix, num_chars, model, state, idx_to_char, char_to_idx):
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]]).view(1, 1)

        Y, state = model(X, state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.squeeze(1).argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])

def init_rnn_state(batch_size, num_hiddens):
    return torch.zeros((1, batch_size, num_hiddens))

def inference_with_RNN(train_chars, test_chars):
    num_hiddens, num_steps = 256, 35
    train_ids, test_ids, word2idx, idx2word = text_processing(train_chars, test_chars, num_steps)
    
    model = Sequence_Modeling(len(word2idx), 300, len(word2idx), num_hiddens) #
    checkpoint = torch.load('./rnn_model.pt')
    model.load_state_dict(checkpoint)

    X = torch.tensor(test_ids, dtype=torch.int)
    state = init_rnn_state(X.size(0), num_hiddens)
    y_hat, _ = model(X, state)
    y_hat = y_hat[:, -1, :].argmax(dim=1).numpy()
    pred_txt = [idx2word[w] for w in y_hat]
    g = open('data/predict.txt', 'w')
    g.write('\n'.join(pred_txt))
    g.close()



def train_with_RNN_easy(train_chars, test_chars):
    batch_size, num_hiddens, num_steps = 64, 256, 35
    train_ids, test_ids, word2idx, idx2word = text_processing(train_chars, test_chars, num_steps)

    model = Sequence_Modeling_pytorch(len(word2idx), 300, len(word2idx), num_hiddens) #
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()

    num_epochs = 1
    for epoch in range(1, num_epochs + 1):
        state = init_rnn_state(batch_size, num_hiddens)
        train_dataloader = data_iter_consecutive(train_ids, batch_size, num_steps)
        train_l_sum, train_acc_sum, n = 0., 0., 0
        for i, Xy in enumerate(train_dataloader):
            state.detach_()
            X, y = Xy
            # print(X.shape, y.shape, state[0].shape)
            y_hat, state = model(X, state)
            # print(y_hat.size(), y.size())
            y_hat = y_hat.view(y_hat.size(0)*y_hat.size(1), -1)
            y = y.view(-1)
            loss = loss_func(y_hat, y.long()).sum()

            # 梯度清零
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_l_sum += loss.item() * y.size(0)
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.size(0)

        print('epoch %d, perplexity %.4f, train acc %.3f'
              % (epoch, math.exp(train_l_sum / n), train_acc_sum / n))

        if epoch % 20 == 0:
            pred_len, prefixes = 50, ['分开', '不分开']
            state = init_rnn_state(1, num_hiddens)
            for prefix in prefixes:
                print(' -', predict_rnn_pytorch(
                    prefix, pred_len, model, state, idx2word, word2idx))

        X = torch.tensor(test_ids, dtype=torch.int)
        state = init_rnn_state(X.size(0), num_hiddens)
        y_hat, _ = model(X, state)
        y_hat = y_hat[:, -1, :].argmax(dim=1).numpy()
        pred_txt = [idx2word[w] for w in y_hat]
        g = open('data/predict.txt', 'w')
        g.write('\n'.join(pred_txt))
        g.close()

if __name__ == '__main__':
    train_chars, test_chars = load_jaychou_lyrics()

    inference_with_RNN(train_chars, test_chars)

    #train_with_RNN_easy(train_chars, test_chars)
