import numpy as np

def compute_acc(pred_file):
    with open('data/test_X.txt') as f:
        gold = f.readlines()
    gold = [sent.strip() for sent in gold]
    gold = [''.join([o for o in reversed(e_idx)]) for e_idx in gold]

    with open(pred_file) as f:
        pred = f.readlines()
    pred = [sent.strip() for sent in pred]
    correct_case = [i for i, _ in enumerate(gold) if gold[i] == pred[i]]

    acc = len(correct_case)*1./len(gold)
    print('The predicted accuracy is %s' %acc)

if __name__ == '__main__':
    pred_file = 'data/predict.txt'
    compute_acc(pred_file)