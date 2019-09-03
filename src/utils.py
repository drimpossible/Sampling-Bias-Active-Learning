from __future__ import absolute_import, division, print_function
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import csv
import logging
import os
import sys
import math

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score


#TODO: Optimize this

def from_fastText_to_csv(path):
    ds = open(path, 'r').readlines()
    y, X = [int(el.split('__label__')[1].split(',')[0]) for el in ds], \
           [el.split('__label__')[1][4:].strip('\n') for el in ds]
    df = pd.DataFrame({'y': y, 'X': X})
    new_path =path + '.csv'
    df.to_csv(new_path, header=False, index=False)
    return new_path


def from_csv_to_fastText(path):
    #TODO will not work for more than 10 classes, fix it
    data = pd.read_csv(path, header=None)
    X, y = data[1], data[0]
    new_path = path.strip('.csv')
    fast_path = to_fastText(X, y, new_path)
    # fast_path = ''.join(fast_path.split('/')[:-1])
    return fast_path

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

def split_data(idx, split_percent):
    np.random.shuffle(idx)
    train_idx = idx[:int(idx.shape[0]*split_percent)]
    pool_idx = idx[int(idx.shape[0]*split_percent):]
    return train_idx, pool_idx


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


def get_labels(labels):
    return np.array([int(label[9:]) for label in labels])


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


def scatter_plot(name_array, y_array, x_array, title, xlabel, ylabel, xlim_min=None, x_lim_max=None,
                 ylim_min=None, ylim_max=None, var=False, std_array=None):
    assert(len(name_array)==len(y_array))
    char_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    linestyles = ['-', '--','-.',':','*',',','+']
    fig = plt.figure(figsize=(6, 5))
    ax = plt.subplot(111)
    for i in range(len(y_array)):
        ax.plot(x_array[i], y_array[i], char_colors[i]+linestyles[i], label=name_array[i], linewidth=2, markersize=7)
        if var:
            plt.fill_between(x_array[i], y_array[i] - std_array[i], y_array[i] + std_array[i], alpha=.1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(linestyle='--')
    plt.xlim([xlim_min, x_lim_max])
    plt.ylim([ylim_min, ylim_max])
    box = ax.get_position()
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1, 0.5))
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    return fig


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
