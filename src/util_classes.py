import copy
import numpy as np
import metrics
import utils


class MetricLogger:
    def __init__(self, opt, logger, mode):
        self.nll_loss = ArrayGenerator('nll_score', 'Epoch')
        self.sph_loss = ArrayGenerator('spherical_loss', 'Epoch')
        self.brier_loss = ArrayGenerator('brier_loss', 'Epoch')
        self.acc = ArrayGenerator('accuracy', 'Epoch')
        self.var_ratio = ArrayGenerator('var_ratio', 'Epoch')
        self.entropy = ArrayGenerator('entropy', 'Epoch')
        self.std = ArrayGenerator('STD', 'Epoch')
        self.ece = ArrayGenerator('ECE', 'Epoch')
        self.refine = ArrayGenerator('refinement', 'Epoch')
        self.brier_multiclass = ArrayGenerator('multiclass_brier_score', 'Epoch')
        self.opt = opt
        self.logger = logger
        self.mode = mode

    def add(self, epoch, prob, labels):
        prob = copy.deepcopy(prob)
        labels = copy.deepcopy(labels)
        nll_m, nll_var = metrics.nll_score(prob, labels)
        ss_m, ss_var = metrics.avg_spherical_score(prob, labels)
        bs_m, bs_var = metrics.brier_score(prob, labels, self.opt.num_classes)
        ece, refine, brier_multi = metrics.ece(prob, labels, n_bins=15)
        acc = metrics.accuracy(prob, labels)
        vr_m, vr_var = metrics.var_ratio(prob)
        ent_m, ent_var = metrics.entropy(prob)
        std_m, std_var = metrics.std_score(prob)
        curr_percent = epoch*self.opt.num_acquise_percent + self.opt.init_train_percent
        self.nll_loss.add(curr_percent, nll_m.item(), nll_var.item())
        self.sph_loss.add(curr_percent, ss_m.item(), ss_var.item())
        self.brier_loss.add(curr_percent, bs_m.item(), bs_var.item())
        self.acc.add(curr_percent, acc.item())
        self.var_ratio.add(curr_percent, vr_m.item(), vr_var.item())
        self.entropy.add(curr_percent, ent_m.item(), ent_var.item())
        self.std.add(curr_percent, std_m.item(), std_var.item())
        self.ece.add(curr_percent, ece)
        self.refine.add(curr_percent, refine.item())
        self.brier_multiclass.add(curr_percent, brier_multi.item())
        self.logger.info('==> {0} Iter: [{1}/{2}] '
                        'Acc:{top1:.4f}\t'
                        'NLL:{nll:.4f}\t'
                        'SphL:{ss:.4f}\t'
                        'BrierL:{bs:.4f}\t'
                        'ECE:{ece_error:.4f}\t'
                        'Ref:{ref_err:.4f}\t'
                        'Amx_Brier:{tot_err:.4f}\t'
                        'VarR:{vr:.4f}\t'
                        'Ent:{ent:.4f}\t'
                        'STD:{std:.4f}\t'.format(self.mode, epoch+1, self.opt.query_iter, top1=acc.item(), nll=nll_m.item(), ss=ss_m.item(),
                                                 bs=bs_m.item(), ece_error=ece.item(), ref_err=refine.item(), tot_err=brier_multi.item(),
                                                 vr=vr_m.item(), ent=ent_m.item(), std=std_m.item()))

    def save(self, path):
        mode = self.mode
        utils.save(path + 'nll_loss_'+mode+'.pkl', self.nll_loss)
        utils.save(path + 'sph_loss_'+mode+'.pkl', self.sph_loss)
        utils.save(path + 'brier_loss_'+mode+'.pkl', self.brier_loss)
        utils.save(path + 'acc_'+mode+'.pkl', self.acc)
        utils.save(path + 'var_ratio_'+mode+'.pkl', self.var_ratio)
        utils.save(path + 'entropy_'+mode+'.pkl', self.entropy)
        utils.save(path + 'std_'+mode+'.pkl', self.std)
        utils.save(path + 'ece_'+mode+'.pkl', self.ece)
        utils.save(path + 'refine_'+mode+'.pkl', self.refine)
        utils.save(path + 'brier_multiclass_'+mode+'.pkl', self.brier_multiclass)


class DataTracker:
    def __init__(self, logger):
        self.acccounter = ArrayTracker('Per sample classification', 'Epoch')
        self.label = ArrayTracker('Label predicted', 'Epoch')
        self.istrain = ArrayTracker('Train pool split', 'Epoch')
        self.logger = logger

    def add(self, epoch, y_prob, y_true, is_train):
        is_train = copy.deepcopy(is_train)
        y_true = copy.deepcopy(y_true)
        y_prob = copy.deepcopy(y_prob)
        acc = metrics.accuracy(y_prob, y_true, return_vec=True)
        y_pred = np.argmax(y_prob, axis=1)
        self.label.add(y_pred, epoch)
        self.acccounter.add(acc, epoch)
        self.istrain.add(is_train, epoch)

    def save(self, path):
        utils.save(path + 'acccounter.pkl', self.acccounter)
        utils.save(path + 'label_counter.pkl', self.label)
        utils.save(path + 'istrain_tracker.pkl', self.istrain)


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum * 1.0 / self.count * 1.0


class ArrayGenerator:
    def __init__(self, yname, xname):
        self.yname = yname
        self.xname = xname
        self.y = []
        self.x = []
        self.y_var = []

    def add(self, x_point, y_point, y_var=-1.0):
        if y_var != -1.0:
            self.y_var.append(y_var)
        self.x.append(x_point)
        self.y.append(y_point)


class ArrayTracker:
    def __init__(self, yname, xname):
        self.yname = yname
        self.xname = xname
        self.arr = []
        self.epoch = []

    def add(self, inp, epoch):
        self.arr.append(inp)
        self.epoch.append(epoch)
