import models
import numpy as np
import utils
import os
import gc


def train(logger, opt, dset, epoch, data_logger):
    logger.info(f"==> Query Iteration:{(epoch+1)}/{opt.query_iter}\tTraining Size:{dset.is_train.sum()}\tPool Size:{(~(dset.is_train)).sum()}")

    if opt.model == 'FastText':
        train_path = dset.generate_data(train=True)
    y_prob_all = np.zeros((len(dset.X),opt.num_classes))*1.0
    y_test = np.zeros((len(dset.X_test),opt.num_classes))*1.0

    for i in range(opt.num_ensemble):
        logger.debug(f"==> Loading model..")
        model = getattr(models, opt.model)(opt)

        logger.debug(f"==> Training model..")
        if opt.model == 'FastText':
            model.fit_(train_path)
            if opt.quantize:
                model.quantize_(train_path)
            logger.debug(f"==> Predicting..")
            y_prob_all += model.predict_proba_(dset.X)
            y_test += model.predict_proba_(dset.X_test)
            y_feat = model.get_features_(dset.X)
        else:
            vectorizer = model.fit_(dset)
            logger.debug(f"==> Predicting..")
            y_prob_all += model.predict_proba_(dset.X, vectorizer)
            y_test += model.predict_proba_(dset.X_test, vectorizer)
            y_feat = model.get_features_(dset.X, vectorizer)
            del vectorizer

        if not os.path.exists(opt.logpath + opt.exp_name + '/pretrained/'): os.makedirs(opt.logpath + opt.exp_name + '/pretrained/')
        model.save_model_(itr=epoch, path=opt.logpath + opt.exp_name + '/pretrained/ensid_'+str(i)+'_', quantized=opt.quantize)
        del model
        gc.collect()

    data_logger.add(epoch, y_prob_all, dset.y, dset.is_train)
    y_prob_all /= (1.0*opt.num_ensemble)
    y_test /= (1.0*opt.num_ensemble)
    return y_prob_all, y_test, y_feat


def acquise(y_prob_all, y_true_all, acq, idx, itr, logger):
    #print(idx)
    y_prob, y_true = y_prob_all[idx], y_true_all[idx]
    #logger.add(itr, y_prob, y_true)
    idx, unc = acq.acquise(y_prob=y_prob)
    return idx, unc


def acquise_coreset(acq, y_feat, num_acq, train_idx, pool_idx):
    idx_batch = acq.acquise(y_feat=y_feat, num_acq=num_acq, train_idx=train_idx, pool_idx=pool_idx)
    return idx_batch