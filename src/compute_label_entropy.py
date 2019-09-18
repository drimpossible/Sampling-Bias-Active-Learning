def compute_label_entropy(lent_path_random, lent_path_unc, logs_dir):
    full_path_random = os.path.join(logs_dir, lent_path_random, 'label_entropy.pkl')
    full_path_unc = os.path.join(logs_dir, lent_path_unc, 'label_entropy.pkl')
    with open(full_path_random, 'rb') as handle: random_lent = pickle.load(handle)
    with open(full_path_unc, 'rb') as handle: unc_lent = pickle.load(handle)
    # print('LE Random :'+ '\n', '$'+str(utils.round_(np.mean(random_lent.y), 2)) + ' \pm ' +
    #       str(utils.round_(np.std(random_lent.y), 2))+'$' + '\n')
    print('LE UNC :'+ '\n', '$'+str(utils.round_(np.mean(unc_lent.y), 4)) + ' \pm ' +
          str(utils.round_(np.std(unc_lent.y), 4))+'$' + '\n')


if __name__ == '__main__':
    import os
    import pickle
    import numpy as np
    import utils

    logs_dir = 'YOUR PATH'

    for model in ['FastText']:
        for ds in ['sogou_news', 'dbpedia',
        'yahoo_answers', 'yelp_review_polarity',
        'yelp_review_full', 'ag_news',
        'amazon_review_polarity', 'amazon_review_full']:
            print('DS IS: ', ds)
            print('Model is: ', model)
            # General Filter
            acc_paths_ds = [el for el in os.listdir(logs_dir) if ds in el]
            acc_paths_model = [el for el in acc_paths_ds if model in el.split('+')]

            # Random Filter
            acc_path_random = [el for el in acc_paths_model if 'random' in el.split('+')]
            acc_path_random_single_seed = [el for el in acc_path_random if 'seed+0' in el]
            acc_path_random_single_seed = [el for el in acc_path_random_single_seed if 'dit' not in el]
            acc_path_random_single_seed_single_run = [el for el in acc_path_random_single_seed if 'run+0' in el]
            acc_path_random_single_seed_single_run_9_iters = [el for el in acc_path_random_single_seed_single_run
                                                               if 'itr+9' in el][0]

            # print(acc_path_random_single_seed_single_run_39_iters.__len__())
            print(acc_path_random_single_seed_single_run_9_iters)

            # Unc Filter
            acc_path_unc = [el for el in acc_paths_model if 'entropy' in el.split('+')]
            acc_path_unc_single_seed = [el for el in acc_path_unc if 'seed+0' in el]
            acc_path_unc_single_seed = [el for el in acc_path_unc_single_seed if 'dit' not in el]
            acc_path_unc_single_seed_single_run = [el for el in acc_path_unc_single_seed if 'run+0' in el]
            acc_path_unc_single_seed_single_run_9_iters = [el for el in acc_path_unc_single_seed_single_run
                                                            if 'itr+9' in el][0]

            # print(acc_path_unc_single_seed_single_run_39_iters.__len__())
            print(acc_path_unc_single_seed_single_run_9_iters)

            compute_label_entropy(acc_path_random_single_seed_single_run_9_iters,
                            acc_path_unc_single_seed_single_run_9_iters,
                            logs_dir=logs_dir)