import logging
import os
import pickle
import random
import numpy as np


def to_fastText(X, y, data_dir, expname, mask=None, mode='train'):
    path = os.path.join(f'{data_dir}{expname}/data_temp.{mode}')
    with open(path, 'w') as f:
        for i in range(y.shape[0]):
            if mask is None:
                f.write('__' + 'label' + '__' + str(y[i]) + ' , ' + X[i] + '\n')
            else:
                if mask[i]:
                    f.write('__' + 'label' + '__' + str(y[i]) + ' , ' + X[i] + '\n')
    return path


def from_fastText(data_dir, mode='train'):
    X, y = [], []
    with open(f'{data_dir}.{mode}', 'r') as f:
        for line in f:
            label = int(line.split(' , ')[0].split('__')[-1].strip())
            assert(isinstance(label, int))
            y.append(label)
            X.append(line[13:].strip('\n'))
    return X, y


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def idx_to_mask(full_size, idx):
    mask = np.full(full_size, False)
    mask[idx] = True
    return mask


def mask_to_idx(mask):
    return np.where(mask)[0]


def get_mask(num_points, num_init):
    mask = np.full(num_points, False)
    arr = np.arange(num_points)
    np.random.shuffle(arr)
    mask[arr[:num_init]] = True
    return mask


def round_(num, dec:int=2):
    return round(num, dec)


def rearrange_label_proba(y_prob, labels):
    labels = np.array(list(map(get_labels, labels)))
    return np.array([subarray[np.argsort(index)] for subarray, index in zip(y_prob, labels)])


def get_logger(folder):
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # file logger
    # folder = os.path.join(os.getcwd(), folder)
    if not os.path.exists(folder):
        os.mkdir(folder)
    fh = logging.FileHandler(os.path.join(folder, 'checkpoint.log'), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def get_labels(labels):
    return np.array([int(label[9:]) for label in labels])


def softmax(x):
    assert(len(x.shape)==2)
    e_x =  np.exp(x*1.0)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div


def load(path):
    with open(path, 'rb') as handle:
        obj = pickle.load(handle)
    return obj

def save(path, obj):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def opt_assert(opt):
    assert(opt.num_ensemble==1 or opt.model == 'FastText')
