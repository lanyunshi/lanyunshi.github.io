import numpy as np

# 题目
# 请在这里实现“多项式基函数”
def multinomial_basis(x, feature_num=10):
    '''多项式基函数'''
    x = np.expand_dims(x, axis=1) # shape(N, 1)
    #==========
    #todo '''请实现多项式基函数'''
    #==========
    ret = None
    return ret

# 答案
def multinomial_basis(x, feature_num=10):
    x = np.expand_dims(x, axis=1) # shape(N, 1)
    feat = [x]
    for i in range(2, feature_num+1):
        feat.append(x**i)
    ret = np.concatenate(feat, axis=1)
    return ret

# 评测脚本
x1 = np.array([1, 2, 3])
y1 = multinomial_basis(x1)
y1 = np.matrix([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
       [3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049]])
x2 = np.array([-1, 0, 0.5])
y2 = multinomial_basis(x2)
y2 = np.matrix([[-1.000000e+00,  1.000000e+00, -1.000000e+00,  1.000000e+00,
        -1.000000e+00,  1.000000e+00, -1.000000e+00,  1.000000e+00,
        -1.000000e+00,  1.000000e+00],
       [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00],
       [ 5.000000e-01,  2.500000e-01,  1.250000e-01,  6.250000e-02,
         3.125000e-02,  1.562500e-02,  7.812500e-03,  3.906250e-03,
         1.953125e-03,  9.765625e-04]])



