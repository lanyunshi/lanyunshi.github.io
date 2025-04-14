import torch
from torch import nn
import math
import string
import random
from rnn_with_atten_solution import Sequence_Modeling

random.seed(2023)

def generate_random_string(string_length):
    """Generate a random string"""

    letters = string.ascii_uppercase
    return ''.join(random.choice(letters) for i in range(string_length))

def get_string_batch(batch_size, length):
    batched_examples = [generate_random_string(length) for _ in range(batch_size)]
    enc_x = [[ord(ch)-ord('A')+1 for ch in list(exp)] for exp in batched_examples]
    y = [[o for o in reversed(e_idx)] for e_idx in enc_x]
    dec_x = [[0]+e_idx[:-1] for e_idx in y]
    return (torch.tensor(enc_x, dtype=torch.int32), \
            torch.tensor(dec_x, dtype=torch.int32), \
            torch.tensor(y, dtype=torch.int32))

def get_test_batch(test_set):
    enc_x = [[ord(ch)-ord('A')+1 for ch in list(exp)] for exp in test_set]
    return torch.tensor(enc_x, dtype=torch.int32).view(1, -1)

def predict_rnn_pytorch(enc_x, model, state):
    output = [0]

    enc_hidden, state = model.encode(enc_x, state)
    for t in range(len(enc_x[0])):
        is_first_step = True if t ==0 else False
        enc_y = torch.tensor([output[-1]]).view(1, 1)
        Y, state = model.decode(enc_y, enc_hidden, state, is_first_step)
        output.append(int(Y.squeeze(1).argmax(dim=1).item()))

    output = ''.join([chr(i+64) for i in output[1:]])

    return output

def init_rnn_state(batch_size, num_hiddens):
    return torch.zeros((1, batch_size, num_hiddens))

def load_file(file_path):
    with open(file_path) as f:
        test_set = f.readlines()

    test_set = [line.strip() for line in test_set]
    return test_set

def train_with_RNN(test_set):
    batch_size, num_hiddens = 64, 100

    word_num = ord('Z')-ord('A')+2
    model = Sequence_Modeling(word_num, 100, word_num, num_hiddens)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    loss_func = nn.CrossEntropyLoss()

    num_epochs = 10000
    for epoch in range(1, num_epochs + 1):
        state = init_rnn_state(batch_size, num_hiddens)
        enc_x, enc_y, y = get_string_batch(batch_size, 8)
        train_l_sum, train_acc_sum, n = 0., 0., 0

        # print(X.shape, y.shape, state[0].shape)
        enc_hidden, state = model.encode(enc_x, state)
        y_hat, _ = model.decode(enc_y, enc_hidden, state)
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

        if epoch % 500  == 0:
            print('epoch %d, perplexity %.4f, train acc %.3f'
                  % (epoch, math.exp(train_l_sum / n), train_acc_sum*1. / n))

            enc_x, enc_y, y = get_string_batch(1, 8)
            state = init_rnn_state(1, num_hiddens)
            y_hat = predict_rnn_pytorch(enc_x, model, state)
            print('predict', ''.join([chr(i+64) for i in y[0]]), y_hat)

    pred_txt = []
    for X in test_set:
        enc_x = get_test_batch(X)
        state = init_rnn_state(1, num_hiddens)
        y_hat = predict_rnn_pytorch(enc_x, model, state)
        pred_txt += [y_hat]
    g = open('data/predict.txt', 'w')
    g.write('\n'.join(pred_txt))
    g.close()



if __name__ == '__main__':
    #print('toy_string', toy_string)
    test_set = load_file('data/test_X.txt')

    train_with_RNN(test_set)

