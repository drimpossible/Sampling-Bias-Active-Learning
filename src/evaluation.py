### 1) Load the metrics stored
### 2) Functions to compute whatever metrics we need
### 3) Add a loop over the scatter plots plotting code
import numpy as np
# from metrics import ArrayGenerator

def compute_forgetting(acc, acc_next):
    return np.logical_and(np.logical_not(acc_next), acc)

# def label_change(y_pred, y_pred_next):
#     return (y_pred_past == y_pred_curr)

# def count_events(acc, label):
#     forgetting = ArrayGenerator('Forgetting Events', 'Epoch')
#     labelchange = ArrayGenerator('Label Change', 'Epoch')
#     forgetting.x = #Input array here
#     forget_pts = np.sum(np.array([compute_forgetting(acc.y[i],acc.y[i+1]) for i in range(acc.x.shape[0]-1)]), axis=0)
#     sorted_fgtpts_idx, sorted_fgtpts = np.argsort(forget_pts), np.sort(forget_pts)
#     forgetting.y = sorted_fgtpts
#     labelchange.x = #Input array here
#     label_pts = np.sum(np.array([label_change(label[i], label[i+1]) for i in range(label.x.shape[0]-1)]), axis=0)
#     sorted_labelchange_idx, sorted_labelchange = np.argsort(label_pts), np.sort(label_pts)
#     labelchange.y = sorted_labelchange
#     #TODO: Calculate spearman correlation between label chnage and forgetting
#
#     with open(metric_logger_pool_path + 'forgetting_points.pkl', 'wb') as handle:
#             pickle.dump(forgetting, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     with open(metric_logger_pool_path + 'labelchange.pkl', 'wb') as handle:
#             pickle.dump(labelchange, handle, protocol=pickle.HIGHEST_PROTOCOL)


# def perform_stability_checks(dset_list):
#     # Count intersection of training points over seeds/models (Section 5.1, 5.2)
#     common_points = ArrayGenerator('Num Intersections', 'Epoch')
#     point_count = np.zeros(dset[0].istrain.shape[0])
#     for dset in dset_list:
#         point_count[dset.istrain] += 1
#     sorted_idx, sorted_pcnt = np.argsort(point_count), np.sort(point_count)
#     common_points.y = sorted_pcnt
#     common_points.x = np.arange(sorted_pcnt.shape[0])
#     common_points.y_var = sorted_idx #Remember this is a n excuse to store these values. Not to be plotted!
#     with open(metric_logger_pool_path + 'stability_checks.pkl', 'wb') as handle:
#             pickle.dump(common_points, handle, protocol=pickle.HIGHEST_PROTOCOL)

def to_fastText(X, y, data_dir, ds, mode):
    path = os.path.join(f'{data_dir}/{ds}_data_temp.{mode}')
    with open(path, 'w') as f:
        for i in range(y.shape[0]):
            f.write('__' + 'label' + '__' + str(y[i]) + ' , ' + X[i] + '\n')
    return path


def from_fastText(path):
    X, y = [], []
    with open(path, 'r') as f:
        for line in f:
            label = int(line.split(' , ')[0].split('__')[-1].strip())
            assert(isinstance(label, int))
            y.append(label)
            X.append(line[13:].strip('\n'))
    return X, y


if __name__ == '__main__':
    import json
    import os
    from fastText import train_supervised
    from sklearn.metrics import accuracy_score
    import pickle
    import pandas as pd

    with open('config.json', 'r') as f:
        config_g = json.load(f)
    data_dir = '/home/verisk/Desktop/Active-Learning-Text/data'
    # data = ['imdb', 'sogou_news', 'dbpedia',
    #         'yahoo_answers', 'yelp_review_polarity',
    #         'ag_news',
    #         'amazon_review_polarity', 'yelp_review_full']
    ds = 'yelp_review_full'
    # for ds in data:
    # ds = 'sogou_news'
    # with open('/media/verisk/Experiments/logs_old/model+FastText+dataset+var_ratio+query_type+'
    #           'margin_score+lr+0.25+epochs+10+itp_nacq+0.0025+itr+39+seed+0+run+0+/istrain_tracker.pkl',
    #           'rb') as handle:
    #     sogou_news_res_samp = pickle.load(handle)
    # res_samp = np.where(sogou_news_res_samp.arr[-1] == True)[0]

    config = config_g['HYPER_PARAMS'][ds]
    # train = pd.read_csv('/home/verisk/Desktop/Active-Learning-Text/data/short_amazon_review_full_csv/'
    #                     'amazon_review_full_short.train', header=None)
    # print('Short train is of size: ', train.shape[0])
    size_train = config['size_train']
    # ratio = train.shape[0] / size_train
    # new_lr = (config['lr']*(1.0 / ratio)) % 1 if (config['lr']*(1.0 / ratio)) % 1 != 0 else 0.75
    new_lr = 0.1
    print('new lr is: ', new_lr)
    # print(f'For {ds} compression is {ratio} of the original dataset' + '\n')
    # full_train = pd.read_csv(f'../data/{ds}_csv/train.csv', header=None)
    # full_train_without_res_samp = full_train.drop(res_samp).reset_index(drop=True)
    # assert len(full_train_without_res_samp) == len(full_train) - len(train)
    # # selected_idx = np.random.permutation(np.arange(len(full_train)))[:len(train)]
    # # random_pow = full_train.loc[selected_idx].reset_index(drop=True)
    # assert size_train == full_train.shape[0]
    # # assert random_pow.shape[0] == train.shape[0]
    # path_train = to_fastText(full_train_without_res_samp[1], full_train_without_res_samp[0], data_dir, ds, mode='train')
    # path_train_random = to_fastText(random_pow[1], random_pow[0], data_dir, ds, mode='rand')
    model_res = train_supervised(dim=config['dim'], input='/home/verisk/Desktop/Active-Learning-Text/data/short_yelp_review_full_csv/yelp_review_full_short.train',
                                               epoch=config['epochs'], lr=new_lr,
                                               wordNgrams=config['ngrams'], verbose=2,
                                            minCount=1, bucket=10000000,
                                               thread=8)
    # model_random = train_supervised(dim=config['dim'], input=path_train_random,
    #                                            epoch=config['epochs'], lr=new_lr,
    #                                            wordNgrams=config['ngrams'], verbose=2,
    #                                         minCount=1, bucket=10000000,
    #                                            thread=8)
    # data_test = pd.read_csv(os.path.join(data_dir, ds_name + '_csv', 'test.csv'), header=None)
    # X_test, y_test = data_test[1], data_test[0]
    test = pd.read_csv(f'../data/{ds}_csv/test.csv', header=None)
    path_test = to_fastText(test[1], test[0], data_dir, ds, mode='test')
    X_test, y_test = from_fastText(path_test)
    raw_pred = model_res.predict(text=list(X_test))[0]
    pred = [int(el[0].split('__')[-1]) for el in raw_pred]
    print(f'Accuracy for resulting sample is {accuracy_score(y_test, pred)}')

    # raw_pred_rand = model_random.predict(text=list(X_test))[0]
    # pred = [int(el[0].split('__')[-1]) for el in raw_pred_rand]
    # print(f'Accuracy for random sample is {accuracy_score(y_test, pred)}')