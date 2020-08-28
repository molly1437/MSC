import random
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

def parse_line(line,task_sign):
    line = line.split("\t")
    label=-1
    if task_sign == 'A':
        if line[2] in ['NOT','not']:
            label = 0
        elif line[2] in ['OFF','off']:
            label = 1
        return line[0], line[1], label
    elif task_sign == 'B':
        if line[3] in ['TIN','tin']:
            label = 0
        elif line[3] in ['UNT','unt']:
            label = 1
        return line[0], line[1], label
    elif task_sign == 'C':
        if line[4] in ['IND','ind']:
            label = 0
        elif line[4] in ['GRP','grp']:
            label = 1
        elif line[4] in ['OTH','oth']:
            label = 2
        return line[0], line[1], label

def read_file(_set,task_sign):
    print(task_sign)
    print('----------------------------------------------------------------------------------------------------------------------------------')
    with open(_set +".csv", "r", encoding="utf-8") as fi:
        lines = fi.read().splitlines() 
        ids, x_train, y_train = zip(*list(map(lambda x: parse_line(x,task_sign), lines[1:])))

        print('Total number of data:',len(x_train))
    return [ {"id":i, "text":x, "label": y} for i, x, y in zip(ids, x_train, y_train) ]

def fold_iterator_sklearn(all_samples, K=10, dev_ratio=0.10, random_seed=1234):
    """yields K tuples of shape (train, dev, test) """
    random.seed(random_seed)
    random.shuffle(all_samples) # initial shuffle
    _all = np.array(all_samples) # convert to numpy for list indexing

    skf = StratifiedKFold(n_splits=K)
    skf.get_n_splits(_all, [ y["label"] for y in _all])

    for train_index, test_index in skf.split(_all, [ y["label"] for y in _all]):
        trn, dev = train_test_split(_all[train_index], test_size=dev_ratio, random_state=random_seed)
        yield (trn, dev, _all[test_index])
    return

