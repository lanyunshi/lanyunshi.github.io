import numpy as np
import random

def load_data(filename):
    """载入数据"""
    xys = []
    with open(filename, 'r') as f:
        for line in f:
            xys.append(map(float, line.strip().split()))

    if 'train' in filename:
        xs, ys = zip(*xys)
        return np.asarray(xs), np.asarray(ys)
    else:
        xs = [list(x)[0] for x in xys]
        return np.asarray(xs), None

class LinearRegression(object):
    def __init__(self):
        super(LinearRegression, self).__init__()
        '''
        self.w --> (2, ) is the parameter of a linear regression
        self.lr is the learning rate of training
        self.epoch is the iteration time of training
        '''
        self.w = 0.05 * np.random.randn(2)
        self.lr = 0.0001
        self.epoch = 1000

    def predict(self, x):
        beta0 = np.expand_dims(np.ones_like(x), axis=1)
        beta1 = np.expand_dims(x, axis=1)
        x = np.concatenate([beta1, beta0], axis=1)

        y = np.dot(x, self.w)
        return y

    def train(self, x, y):
        '''
        x and y are the data for traning a linear regression
        please simply update the value of self.w and not include any other parameters
        '''

        # ==========
        # todo '''使用随机梯度下降法优化对self.w进行更新'''


        # ==========

    def LSE(self, x, y):
        '''
        x and y are the data for estimate a linear regression
        '''

        # ==========
        # todo '''使用最小二乘法对self.w进行估计'''


        # ==========



def evaluate(ys, ys_pred):
    """评估模型。"""
    std = np.sqrt(np.mean(np.abs(ys - ys_pred) ** 2))
    return std

if __name__ == '__main__':
    solver = 'GD'
    train_file = './input/train.txt'
    test_file = './input/test_X.txt'
    # 载入数据
    x_train, y_train = load_data(train_file)
    x_test, _ = load_data(test_file)

    # 使用线性回归训练模型，返回一个函数f()使得y = f(x)
    f = LinearRegression()
    
    if solver == 'GD':
        f.train(x_train, y_train)
    elif solver == 'LSE':
        f.LSE(x_train, y_train)
    else:
        raise TypeError("Wrong solver !")
    y_train_pred = f.predict(x_train)
    std = evaluate(y_train, y_train_pred)
    print('The std on training data via SGD is ：{:f}'.format(std))

    preds = f.predict(x_test)
    np.save('./output/predict', preds)
