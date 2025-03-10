import numpy as np
import gzip

def compute_acc(pred_file):
    with gzip.open(r'.\data\t10k-labels.gz', 'rb') as f:
        gold = np.frombuffer(f.read(), np.uint8, offset=8)

    with open(pred_file) as f:
        pred = f.readlines()
    pred = [int(sent.strip()) for sent in pred]
    correct_case = [i for i, _ in enumerate(gold) if gold[i] == pred[i]]

    acc = len(correct_case)*1./len(gold)
    print('The predicted accuracy is %s' %acc)

if __name__ == '__main__':
    pred_file = 'data/predict.txt'
    compute_acc(pred_file)