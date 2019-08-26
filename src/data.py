import utils
from sklearn import preprocessing
import numpy as np


class TextData():
    def __init__(self, opt):
        self.opt = opt
        self.X, self.y = utils.from_fastText(opt.data_dir + opt.dataset, mode='train')
        self.X_test, self.y_test = utils.from_fastText(opt.data_dir + opt.dataset, mode='test')
        self.y, self.y_test = np.array(self.y), np.array(self.y_test)
        self.label_enc = preprocessing.LabelEncoder()
        self.y = self.label_enc.fit_transform(self.y)
        self.y_test = self.label_enc.transform(self.y_test)

        self.ohe_enc = preprocessing.LabelBinarizer()
        self.ohe_enc.fit(self.y)
        opt.num_classes = self.ohe_enc.classes_.shape[0]
        opt.num_points = self.y.shape[0]
        opt.num_test = self.y_test.shape[0]
        equal_labels = np.ones(opt.num_classes)*(1.0/opt.num_classes)
        opt.best_label_entropy = -np.sum(equal_labels * np.log(equal_labels+ np.finfo(float).eps))
        num_init = int(opt.init_train_percent * opt.num_points)
        self.is_train = utils.get_mask(opt.num_points, num_init)

    def generate_data(self, train=True):
        mask = self.is_train if train else ~self.is_train
        if train:
            file_path = utils.to_fastText(X=self.X, y=self.y, data_dir=self.opt.logpath, expname=self.opt.exp_name, mask=mask, mode='train')
        else:
            file_path = utils.to_fastText(X=self.X, y=self.y, data_dir=self.opt.logpath, expname=self.opt.exp_name, mask=mask, mode='pool')
        return file_path
    
    def get_X_y(self, train=True):
        mask = self.is_train if train else ~self.is_train
        X = [self.X[i] for i in range(mask.shape[0]) if mask[i]]
        y = self.y[mask]
        return X, y