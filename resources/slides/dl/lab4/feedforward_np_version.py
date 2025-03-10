import numpy as np

class Matmul:
    def __init__(self):
        self.mem = {}

    def forward(self, W, x):
        h = np.matmul(W, x)
        self.mem = {'x': x, 'W': W}
        return h

    def backward(self, grad_y):
        '''
        x: shape(d, N)
        w: shape(d', d)
        grad_y: shape(d', N)
        '''
        x = self.mem['x']
        W = self.mem['W']

        grad_x = np.matmul(W.T, grad_y)
        grad_W = np.matmul(grad_y, x.T)
        return grad_x, grad_W


class Relu:
    def __init__(self):
        self.mem = {}

    def forward(self, x):
        self.mem['x'] = x
        return np.where(x > 0, x, np.zeros_like(x))

    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''

        # ==========
        # todo '''请完成激活函数的梯度后传'''
        # ==========


class Softmax:
    '''
    softmax over last dimention
    '''

    def __init__(self):
        self.epsilon = 1.e-8
        self.mem = {}

    def forward(self, x):
        '''
        x: shape(N, c)
        '''
        x_exp = np.exp(x)
        partition = np.sum(x_exp, axis=1, keepdims=True)
        out = x_exp / (partition + self.epsilon)
        #print(x_exp[:3, :3], out[:3, :3])

        self.mem['out'] = out
        self.mem['x_exp'] = x_exp
        return out

    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        # ==========
        # todo '''请完成Softmax函数的梯度后传'''
        # ==========


class Log:
    '''
    softmax over last dimention
    '''

    def __init__(self):
        self.epsilon = 1e-12
        self.mem = {}

    def forward(self, x):
        '''
        x: shape(N, c)
        '''
        out = np.log(x + self.epsilon)

        self.mem['x'] = x
        return out

    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        x = self.mem['x']

        return 1. / (x + 1e-12) * grad_y


class Model_NP:
    def __init__(self, num_inputs, num_outputs, num_hiddens = 100, lr = 5.e-5, lambda1 = 0.01):
        self.W1 = np.random.normal(size=[num_hiddens, num_inputs + 1])
        self.W2 = np.random.normal(size=[num_outputs, num_hiddens])

        self.mul_h1 = Matmul()
        self.mul_h2 = Matmul()
        self.relu = Relu()
        self.softmax = Softmax()
        self.log = Log()

        self.lr = lr
        self.lambda1 = lambda1 # Lambda is the coefficient of the L2 regularization

    def compute_loss(self, log_prob, labels):
        '''
        log_prob is the predicted probabilities
        labels is the ground truth
        Please return the loss
        '''

        # ==========
        # todo '''请完成多分类问题的损失计算 损失为： 交叉熵损失 + L2正则项'''
        # ==========


    def forward(self, x):
        '''
        x is the input features
        Please return the predicted probabilities of x
        '''

        # ==========
        # todo '''请搭建一个MLP前馈神经网络 补全它的前向传播 MLP结构为FFN --> RELU --> FFN --> Softmax'''
        # ==========

    def backward(self, label):
        '''
        label is the ground truth
        Please compute the gradients of self.W1 and self.W2
        '''

        # ==========
        # todo '''补全该前馈神经网络的后向传播算法'''
        # ==========


    def update(self):
        '''
        Please update self.W1 and self.W2
        '''

        # ==========
        # todo '''更新该前馈神经网络的参数'''
        # ==========



if __name__ == '__main__':
    model = Model_NP()