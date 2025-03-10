import torch
from torch import nn

class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

class Model_Pytorch(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens = 100):
        super(Model_Pytorch, self).__init__()
        self.net = nn.Sequential(
            FlattenLayer(),
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens, num_outputs),
        )

        self.activate_func = nn.Softmax(dim=1)

        print('model with pytorch ...')

    def forward(self, X):
        z = self.net(X)
        y = self.activate_func(z)

        return y

if __name__ == '__main__':
    model = Model_NP()