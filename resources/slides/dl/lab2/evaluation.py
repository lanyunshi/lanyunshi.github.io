import numpy as np
import gzip

def compute_acc():
    with open(r'.\input\test_y.txt', 'r') as f:
        gold = f.readlines()
    ys = [float(x.strip()) for x in gold]

    ys_pred = np.load('./output/predict.npy')
    std = np.sqrt(np.mean(np.abs(np.asarray(ys) - np.asarray(ys_pred)) ** 2))

    print('The std on test data is %f' %std)

if __name__ == '__main__':
    compute_acc()