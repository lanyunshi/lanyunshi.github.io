import sys
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

#print(sys.path)
np.random.seed(2022)
torch.manual_seed(2022)


def softmax(X):
    '''
    X is the input 
    Please compute its softmax outputs
    '''
    # ===============
    '''不调用torch的softmax，手写softmax函数，并返回'''

    # ===============
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition

class SoftmaxRegression(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs, dtype=torch.float64)

        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        nn.init.constant_(self.linear.bias, val=0)

        self.lr = 0.01
        self.num_classes = num_outputs

    def forward(self, x):
        y = self.linear(x.view(x.shape[0], -1))
        return y

    def manual_backward(self, X, y, y_hat):
        '''
        X is the input feature;
        y is the ground truth label;
        y_hat is the predicted label.
        Please update self.linear.weight and self.linear.bias
        '''
        with torch.no_grad():
            # ===============
            '''将automatic_update设为False，不调用torch的自动更新，手动完成梯度下降法优化'''

            # ===============
            y_onehot = torch.nn.functional.one_hot(y.long(), self.num_classes)
            delta_w = - torch.matmul(torch.t(X), (y_hat - y_onehot))/y.size(0)
            self.linear.weight += self.lr * torch.t(delta_w)
            delta_b = - torch.sum((y_hat - y_onehot), dim=0)/y.size(0)
            self.linear.bias += self.lr * delta_b

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


def train_model(train_set, automatic_update=False):
    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)

    #print('train_iter', len(train))
    model = SoftmaxRegression(2, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_func = nn.CrossEntropyLoss()

    num_epochs = 20
    animation_fram = []
    for epoch in range(1, num_epochs + 1):
        train_l_sum, train_acc_sum, n = 0., 0., 0
        for Xy in train_dataloader:
            Xy = Xy.squeeze(1)
            X, y = Xy[:, :-1], Xy[:, -1]
            y_hat = model(X).squeeze(1)

            if automatic_update:
                loss = loss_func(y_hat, y.long()).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            else:
                loss = loss_func(y_hat, y.long()).sum()
                y_hat = softmax(Y-hat)
                model.manual_backward(X, y, y_hat)


            train_l_sum += loss.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
            animation_fram.append((model.linear.weight.detach().numpy()[0, 0], \
                                   model.linear.weight.detach().numpy()[0, 1], \
                                   model.linear.bias.detach().numpy(), loss.detach().numpy()))

        print('epoch %d, loss %.4f, train acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n))
        # test_acc = evaluate_accuracy(test_iter, model)
        # print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
        #       % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

    return model

def load_data(filename):
    """载入数据。"""
    xys = []
    with open(filename, 'r') as f:
        for line in f:
            xys.append(line.strip().split())

    if 'train' in filename:
        C1 = [list(map(float, x)) for x in xys if float(list(x)[2]) == 0]
        C2 = [list(map(float, x)) for x in xys if float(list(x)[2]) == 1]
        C3 = [list(map(float, x)) for x in xys if float(list(x)[2]) == 2]
        return np.asmatrix(C1), np.asmatrix(C2), np.asmatrix(C3)
    else:
        xs = [list(map(float, x)) for x in xys]
        return np.asmatrix(xs), None

if __name__ == '__main__':
    train_file = './input/train.txt'
    test_file = './input/test_X.txt'
    # 载入数据
    C1, C2, C3 = load_data(train_file)
    C, _ = load_data(test_file)

    train_set = np.concatenate((C1, C2, C3), axis=0)
    test_dataloader = DataLoader(C, batch_size=10000, shuffle=False)

    # train model using data set and output animation frame
    my_model = train_model(train_set)

    test_set = next(iter(test_dataloader)).squeeze(1)
    Z = my_model(test_set).detach().numpy()
    preds = np.argmax(Z, axis=1)
    np.save('./output/predict', preds)

    # generate animation
    plt.scatter(np.array(C1[:, 0]), np.array(C1[:, 1]), c='b', marker='+')
    plt.scatter(np.array(C2[:, 0]), np.array(C2[:, 1]), c='g', marker='o')
    plt.scatter(np.array(C3[:, 0]), np.array(C3[:, 1]), c='r', marker='*')

    x = np.arange(0., 10., 0.1)
    y = np.arange(0., 10., 0.1)

    X, Y = np.meshgrid(x, y)
    inp = np.array(list(zip(X.reshape(-1), Y.reshape(-1))), dtype=np.float64)
    #print(inp[:100])
    dataloader = DataLoader(inp, batch_size=10000, shuffle=False)
    inp = next(iter(dataloader))
    Z = my_model(inp).detach().numpy()
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(X.shape)
    print(Z.shape)
    plt.contour(X, Y, Z)
    plt.show()
