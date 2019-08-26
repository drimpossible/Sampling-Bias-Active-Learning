import os
import gc
import opts
import utils
import models
import numpy as np
import util_classes
import metrics
import data
import query_methods
import pickle
import trainer
import matplotlib.pyplot as plt

if __name__ == '__main__':
    global opt, best_prec1
    parser = opts.myargparser()
    opt = parser.parse_args()
    utils.opt_assert(opt=opt)
    utils.seed_everything(seed=opt.seed)

    if not os.path.exists(opt.logpath + opt.exp_name + '/logger/'): os.makedirs(opt.logpath + opt.exp_name + '/logger/')
    logger = utils.get_logger(opt.logpath + opt.exp_name + '/logger/')
    logger.debug(f"==> Starting experiment..")


    logger.debug(f"==> Initializing data..")
    dset = data.TextData(opt)
    num_acq, num_del = int(opt.num_acquise_percent * opt.num_points), 0
    if opt.num_delete_percent is not None:
        num_del = int(opt.num_delete_percent * opt.num_points)

    logger.debug(f"==> Initializing data loggers..")
    metric_logger_train, metric_logger_pool, metric_logger_test = util_classes.MetricLogger(opt, logger, mode='Train'), \
                                                                  util_classes.MetricLogger(opt, logger, mode='Pool'), \
                                                                  util_classes.MetricLogger(opt, logger, mode='Test')
    data_logger = util_classes.DataTracker(logger)
    label_entropy = util_classes.ArrayGenerator('Acquised_Label_Entropy', 'Epoch')
    logger.info(f"==> Hyperparamaters: " + str(opt))

    if opt.expmode == 'train':
        acq = query_methods.AcquisePoints(opt)

        for itr in range(opt.query_iter):
            best_prec1 = 0.0
            y_prob_all, y_prob_test, y_feat = trainer.train(opt=opt, dset=dset, epoch=itr, logger=logger, data_logger=data_logger)

            pool_idx = np.array(utils.mask_to_idx(~dset.is_train))
            train_idx = np.array(utils.mask_to_idx(dset.is_train))
            #ax.scatter(x=y_feat[train_idx[:1000],0], y=y_feat[train_idx[:1000],1],  c=dset.y[train_idx[:20]], alpha=.7, cmap='tab10')
            assert(np.intersect1d(pool_idx, train_idx).shape[0] == 0)
            assert(pool_idx.shape[0]+train_idx.shape[0] == dset.y.shape[0])

            logger.debug(f"==> Adding samples..")
            if not opt.query_type == 'coreset':
                print(y_prob_all.shape)
                idx, unc = trainer.acquise(y_prob_all=y_prob_all, y_true_all=dset.y, acq=acq, idx=pool_idx, itr=itr, logger=metric_logger_pool)
                acquised_idx, acquised_funcvals = idx[:num_acq + num_del], unc[:num_acq + num_del]
            else:
                acquised_idx, dist = trainer.acquise_coreset(acq=acq, y_feat=y_feat, num_acq=num_acq, train_idx=train_idx, pool_idx=pool_idx)
                assert(len(acquised_idx) == num_acq)

            assert(~np.any(dset.is_train[pool_idx[acquised_idx]]))
            dset.is_train[pool_idx[acquised_idx]] = True

            l_ent = metrics.label_entropy(dset.ohe_enc, dset.y[pool_idx[acquised_idx]], is_binary=True) if \
                opt.num_classes == 2 else metrics.label_entropy(dset.ohe_enc, dset.y[pool_idx[acquised_idx]])
            label_entropy.add(itr, l_ent.item())
            logger.info(f"==> Label Entropy: {l_ent:.4f}")

            if opt.num_delete_percent is not None:
                logger.debug(f"==> Removing samples..")
                idx, unc = trainer.acquise(y_prob_all, dset.y, acq, train_idx, itr, metric_logger_train)
                idx, unc = np.flip(idx), np.flip(unc)
                deleted_idx, deleted_funcvals = idx[:num_del], unc[:num_del]
                assert(np.all(dset.is_train[train_idx[deleted_idx]]))
                dset.is_train[train_idx[deleted_idx]] = False

            metric_logger_test.add(itr, y_prob_test, dset.y_test)
            del y_prob_all, y_prob_test, pool_idx, train_idx
            gc.collect()

        logger.debug(f"==> Saving generated metrics..")
        logger_path = f'{opt.logpath + opt.exp_name}/'
        if not os.path.exists(logger_path): os.makedirs(logger_path)
        data_logger.save(logger_path)
        metric_logger_train.save(logger_path)
        metric_logger_pool.save(logger_path)
        metric_logger_test.save(logger_path)
        with open(logger_path + 'label_entropy.pkl', 'wb') as handle:
            pickle.dump(label_entropy, handle, protocol=pickle.HIGHEST_PROTOCOL)
        utils.to_fastText(['Done'], np.array([-1]), opt.logpath, opt.exp_name, mode='train')
        logger.debug(f"==> Experiment completed!")

    elif opt.expmode == 'replicate':
        istrain_tracker = utils.load(opt.expload+'istrain_tracker.pkl')
        dset.is_train = np.logical_and(istrain_tracker.arr[-1],~istrain_tracker.arr[1])
        pool_idx = np.array(utils.mask_to_idx(~dset.is_train))
        train_idx = np.array(utils.mask_to_idx(dset.is_train))
        y_prob_all, y_prob_test, y_feat = trainer.train(opt=opt, dset=dset, epoch=1, logger=logger, data_logger=data_logger)
        metric_logger_test.add(1, y_prob_test, dset.y_test)
