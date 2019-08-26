import metrics
import numpy as np

class AcquisePoints():
    def __init__(self, opt):
        self.query_type = opt.query_type
        self.uncertainty, self.diversity = False, False
        if self.query_type in ['entropy', 'var_ratio', 'std_score', 'margin_score', 'random']:
            self.uncertainty = True
        if self.query_type in ['coreset']:
            self.diversity = True

    def acquise(self, y_prob=None, y_feat=None, train_idx=None, pool_idx=None, num_acq=None):
        if self.uncertainty:
            uncertainty = getattr(metrics, self.query_type)(y_prob, return_vec=True)
            sorted_idx, sorted_uncertainty = np.argsort(uncertainty), np.sort(uncertainty)
            if self.query_type in ['entropy', 'var_ratio']:
                sorted_idx, sorted_uncertainty = np.flip(sorted_idx), np.flip(sorted_uncertainty)
            return sorted_idx, sorted_uncertainty
        if self.diversity:
            acquised_idx, maxmin_dist = metrics.get_coreset(y_feat=y_feat, train_idx=train_idx, pool_idx=pool_idx, query_size=num_acq)
            assert(np.unique(acquised_idx).shape[0]==acquised_idx.shape[0])
            return acquised_idx, maxmin_dist