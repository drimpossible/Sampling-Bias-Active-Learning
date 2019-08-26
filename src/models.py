import fastText
import numpy as np
import torch
import utils
import sys
import random
import os
import pickle
import logging
import json
from tqdm import tqdm, trange
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class FastText(fastText.FastText._FastText):
    def __init__(self, opt):
        super(FastText, self).__init__()
        self.opt = opt
        self.model = None

    def fit_(self, train_path):
        self.model = fastText.train_supervised(dim=self.opt.dim, input=train_path, epoch=self.opt.num_epochs,
                                            lr=self.opt.lr, wordNgrams=self.opt.num_ngrams, verbose=0,
                                            minCount=self.opt.min_count, bucket=self.opt.num_buckets,
                                               thread=self.opt.workers)

    def quantize_(self, train_path):
        self.model.quantize(input=train_path, qnorm=self.opt.qnorm, retrain=self.opt.retrain_quantize,
                            cutoff=self.opt.cutoff, qout=self.opt.qout, thread=self.opt.workers)

    def get_features_(self, X):
        return np.array(list(map(self.model.get_sentence_vector, X)))

    def predict_proba_(self, X):
        y_label, y_proba = self.model.predict(text=X, k=self.opt.num_classes)
        return utils.rearrange_label_proba(y_proba, y_label)

    def save_model_(self, path, itr, quantized=True):
        if quantized:
            self.model.save_model(path + f'fasttext_{itr}.ftz')
        else:
            self.model.save_model(path + f'fasttext_{itr}.bin')

    def load_model_(self, path):
        self.model = fastText.load_model(path)


class NaiveBayes:
    def __init__(self, opt):
        self.opt = opt
        self.model = MultinomialNB()

    def fit_(self, dset):
        X, y = dset.get_X_y(train=True)
        vectorizer = TfidfVectorizer(lowercase=True, max_features=50000, stop_words='english', sublinear_tf=True).fit(X)
        X = vectorizer.transform(X)
        self.model.fit(X, y)
        return vectorizer

    def get_features_(self, X, vectorizer):
        return vectorizer.transform(X)

    def predict_proba_(self, X, vectorizer):
        return self.model.predict_proba(vectorizer.transform(X))

    def save_model_(self, path, itr, quantized=True):
        with open(path + f'naivebayes_{itr}.bin', 'wb') as file:
            pickle.dump(self.model, file)


class LinearSVM:
    def __init__(self, opt):
        self.opt = opt
        svm = LinearSVC()
        self.model = CalibratedClassifierCV(svm, cv=5, method='isotonic')

    def fit_(self, dset):
        X, y = dset.get_X_y(train=True)
        vectorizer = TfidfVectorizer(lowercase=True, max_features=50000, stop_words='english', sublinear_tf=True).fit(X)
        X = vectorizer.transform(X)
        self.model.fit(X, y)
        return vectorizer

    def get_features_(self, X, vectorizer):
        return vectorizer.transform(X)

    def predict_proba_(self, X, vectorizer):
        return self.model.predict_proba(vectorizer.transform(X))

    def save_model_(self, path, itr, quantized=True):
        with open(path + f'linearsvm_{itr}.bin', 'wb') as file:
            pickle.dump(self.model, file)
