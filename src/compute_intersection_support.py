import copy
import matplotlib.pyplot as plt


def clean_from_iter(all_arrays, iter):
    cleaned_arrays = []
    for array in all_arrays:
        to_false = np.where(array.arr[iter] == True)[0]
        for el in array.arr:
            for idx in to_false:
                el[idx] = False
        cleaned_arrays.append(array.arr)
    return cleaned_arrays


def remove_false_from_iter(array, idxs):
    for id in idxs:
        array[id] = False
    return array


def get_intersection_precent_across_time(*arrays, substracting=False, ds='', plot=True):
    all_arrays = []
    for path in arrays:
        full_path = os.path.join(os.path.join(args.logs_dir, path, 'istrain_tracker.pkl'))
        with open(full_path, 'rb') as handle: all_arrays.append(pickle.load(handle))

    if not substracting:
        cleaned_arrays = clean_from_iter(all_arrays, 0)
        series_res = []
        for a1, a2, a3 in zip(cleaned_arrays[0], cleaned_arrays[1], cleaned_arrays[2]):
            final_array = np.logical_and.reduce((a1, a2, a3))
            res = final_array.sum() / a1.sum()
            res = utils.round_(res * 100)
            series_res.append(res)
            print(res)
            print(final_array.sum(), a1.sum())
        if plot:
            plt.plot(np.arange(len(series_res)), series_res)
            plt.title(str(ds))
            plt.show()
        return series_res
    else:
        series_res = []
        res_1, res_2, res_3 = all_arrays[0].arr[-1], all_arrays[1].arr[-1], all_arrays[2].arr[-1]
        arrays_seed_0, arrays_seed_1, arrays_seed_2 = all_arrays[0].arr, all_arrays[1].arr, all_arrays[2].arr
        for i in range(len(all_arrays[0].arr) - 1):
            to_false_1, to_false_2, to_false_3 = np.where(arrays_seed_0[i] == True)[0], \
                                                  np.where(arrays_seed_1[i] == True)[0], \
                                                  np.where(arrays_seed_2[i] == True)[0]

            new_true = copy.deepcopy(arrays_seed_0[i+1])
            new_true[to_false_1] = False

            final_array = np.logical_and.reduce((new_true, res_2, res_3))
            print(final_array.sum(), new_true.sum(), arrays_seed_0[i].sum(), arrays_seed_0[i+1].sum())
            res = final_array.sum() / new_true.sum()
            res = utils.round_(res * 100)
            print(res)
            series_res.append(res)
        if plot:
            fig = plt.figure()
            plt.plot(np.arange(len(series_res)), series_res)
            plt.title(str(ds))
            fig.savefig(f'../plots_stopping/temp_{ds}.png')
        return series_res


def get_intersection_percent_(*arrays):
    all_arrays = []
    for path in arrays:
        full_path = os.path.join(os.path.join(logs_dir, path, 'istrain_tracker.pkl'))
        with open(full_path, 'rb') as handle:all_arrays.append(pickle.load(handle))
    res_samps = []
    for array in all_arrays:
        to_false = np.where(array.arr[0] == True)[0]
        res_samp = array.arr[-1]
        for idx in to_false:
            res_samp[idx] = False
        res_samps.append(res_samp)

    final_array = np.logical_and.reduce(res_samps)
    res = final_array.sum() / res_samps[0].sum()
    # print(final_array.sum(), res_samps[0].sum())
    res = utils.round_(res * 100)
    print(str(res))
    return res, final_array


def to_fastText(X, y, mode='train', ds_name=''):
    y, X = list(y), list(X)
    path = os.path.join(f'../data/{ds_name}_{mode}.train')
    with open(path, 'w') as f:
        for i in range(len(y)):
            f.write('__' + 'label' + '__' + str(y[i]) + ' , ' + X[i] + '\n')
    return path


def creating_pow_set(final_array, ds='ds_name'):
    train = pd.read_csv(f'../data/{ds}_csv/train.csv', header=None)
    to_be_selected = np.where(final_array == True)[0]
    power_set = train.loc[to_be_selected]
    random_power_set = train.sample(len(power_set))
    assert len(power_set) == len(random_power_set) == len(to_be_selected)
    to_fastText(power_set[1], power_set[0], mode='pow', ds_name=ds)
    to_fastText(random_power_set[1], random_power_set[0], mode='random_pow', ds_name=ds)
    power_set.to_csv(f'../data/{ds}_csv_pow/train.csv', header=False, index=False)
    random_power_set.to_csv(f'../data/{ds}_csv_random_pow/train.csv', header=False, index=False)


def intersection_support_vectors(path_to_support_vectors, *paths_to_runs):
    with open(path_to_support_vectors, 'rb') as handle:
        support_vectors = pickle.load(handle)
    print(f'There are {len(support_vectors)} support vectors')
    all_arrays = []
    for path in paths_to_runs[0]:
        full_path = os.path.join(os.path.join(args.logs_dir, path, 'istrain_tracker.pkl'))
        with open(full_path, 'rb') as handle: all_arrays.append(pickle.load(handle))
    res_samps = []
    for array in all_arrays:
        to_false = np.where(array.arr[0] == True)[0]
        res_samp = array.arr[-1]
        for idx in to_false:
            res_samp[idx] = False
        res_samps.append(res_samp)

    # final_array = np.logical_and.reduce(res_samps)
    intersections_unc, intersections_random = [], []
    for res in res_samps:

        # Uncertainty average
        idx_res = np.where(res == True)[0]
        print(f'res samp size {len(idx_res)}')

        count = 0
        for el in support_vectors:
            if el in idx_res:
                count += 1
        result = utils.round_((1.0 * count) / (1.0 * len(support_vectors))) * 100
        intersections_unc.append(result)

        # Random average
        rand = np.arange(res.shape[0])
        np.random.shuffle(rand)
        rand = rand[:idx_res.shape[0]]
        count = 0
        for el in support_vectors:
            if el in rand:
                count += 1
        result = utils.round_((1.0 * count) / (1.0 * len(support_vectors))) * 100
        intersections_random.append(result)

    print(f'For UNC {utils.round_(np.mean(intersections_unc))} \pm {utils.round_(np.std(intersections_unc))}  '
              f'of the support vectors are in the final array')

    print(f'For Random {utils.round_(np.mean(intersections_random))} \pm {utils.round_(np.std(intersections_random))}  '
          f'of the support vectors are in the final array')


def assert_iter_paths(paths, iter):
    for el in paths:
        assert iter in el


if __name__ == '__main__':
    import os
    import pickle
    import numpy as np
    import utils
    import pandas as pd

    import argparse

    # We are assuming two runs and three seeds with 39 iterations, no ensemble or deletion and entropy acquisition function
    parser = argparse.ArgumentParser(description='Intersection Resulting Samples')
    parser.add_argument('--logs_dir', type=str, default='../logs/', help='Logs directory')
    parser.add_argument('--model', type=str, default='FastText', help='Model used')
    parser.add_argument('--dataset', type=str, default='ag_news', help='Dataset used')
    parser.add_argument('--same_seed', type=bool, help='Same Seed, Different Runs')
    parser.add_argument('--dif_seed', type=bool, help='Same Run, Different Seeds')

    args = parser.parse_args()
    for i in ['0.1', '10', '100', '1000']:
        path_to_support = f'../support_vectors/{args.dataset}_C{i}/{args.dataset}_supports.pkl'
        print(path_to_support)
        paths_ds = [el for el in os.listdir(args.logs_dir) if args.dataset in el]
        paths_model = [el for el in paths_ds if args.model in el.split('+')]
        path_random = [el for el in paths_model if 'entropy' in el.split('+')]
        path_random = [el for el in path_random if 'dit' not in el]
        path_random = [el for el in path_random if 'num_ensemble' not in el]
        path_run_0 = sorted([el for el in path_random if 'run+0' in el])
        path_seed_0_run_0, path_seed_1_run_0, path_seed_2_run_0 = [el for el in path_run_0 if 'seed+0' in el][0], \
                                                                  [el for el in path_run_0 if 'seed+1' in el][0], \
                                                                  [el for el in path_run_0 if 'seed+2' in el][0]

        # Filter
        assert_iter_paths([path_seed_0_run_0, path_seed_1_run_0, path_seed_2_run_0], 'itr+39')
        assert_iter_paths([path_seed_0_run_0, path_seed_1_run_0, path_seed_2_run_0], 'entropy')
        intersection_support_vectors(path_to_support, [path_seed_0_run_0, path_seed_1_run_0, path_seed_2_run_0])

