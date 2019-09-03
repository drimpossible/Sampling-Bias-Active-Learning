from sklearn.metrics import calinski_harabaz_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
import utils
from sklearn import preprocessing
#from numba import njit, prange
import gc

#@njit(parallel=True,fastmath=True)
def calculate_distance(selected_set, unselected_set):
    tot = np.zeros(unselected_set.shape[0])
    for i in range(unselected_set.shape[0]):
        dist = np.zeros(selected_set.shape[0])
        for j in prange(selected_set.shape[0]):
            dist[j] = np.sqrt(np.square(unselected_set[i,:]-selected_set[j,:]).sum()+2.0e-16)
        tot[i] = np.min(dist)
    return tot

def get_coreset(y_feat, train_idx, pool_idx, query_size):
    acq_idx, minmax_dist = [], []
    y_pool = y_feat[pool_idx]
    y_train = y_feat[train_idx]
    min_distances = calculate_distance(unselected_set=y_pool, selected_set=y_train)
    assert(np.isfinite(min_distances).all())
    idx, dis = np.argmax(min_distances), np.max(min_distances)
    minmax_dist.append(dis)
    acq_idx.append(idx)
    gc.collect()

    for _ in range(query_size-1):
        #print(y_pool.shape,y_feat[idx].reshape(1,-1).shape)
        dist = calculate_distance(unselected_set=y_pool, selected_set=y_feat[pool_idx[idx].reshape(1,-1)])
        min_distances = np.minimum(min_distances, dist)
        assert(np.isfinite(min_distances).all())
        idx, dis = np.argmax(min_distances), np.max(min_distances)
        minmax_dist.append(dis)
        acq_idx.append(idx)
        gc.collect()
    #print(acq_idx,minmax_dist)
    return np.array(acq_idx), np.array(minmax_dist)

def accuracy(y_prob, y_true, return_vec=False):
    y_pred = np.argmax(y_prob, axis=1)
    arr = (y_pred == y_true)
    if return_vec:
        return arr
    return arr.mean()


def f1(y_prob, y_true):
    y_pred = np.argmax(y_prob, axis=1)
    return f1_score(y_true, y_pred, average='macro')


def conf_matrix(y_prob, y_true):
    y_pred = np.argmax(y_prob, axis=1)
    return confusion_matrix(y_true, y_pred)


def ece(y_prob, y_true, n_bins):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidences, predictions = np.max(y_prob, axis=1), np.argmax(y_prob, axis=1)
    accuracies = np.equal(predictions, y_true)
    ece, refinement = 0.0, 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = np.greater(confidences, bin_lower) * np.less_equal(confidences, bin_upper)
        prop_in_bin = np.mean(in_bin*1.0)
        if prop_in_bin.item() > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.absolute(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            refinement += (accuracy_in_bin * (1 - accuracy_in_bin)) * prop_in_bin
            refinement += (accuracy_in_bin * (1-accuracy_in_bin)) * prop_in_bin

    total = ece + refinement
    return ece, refinement, total


# Uncertainty Metrics

def var_ratio(y_prob, return_vec=False):
    ratio = np.max(y_prob, axis=1)
    ratio = 1.0 - ratio
    var_ratio_mean = np.mean(ratio)
    var_ratio_var = np.var(ratio)
    if return_vec:
        return ratio
    return var_ratio_mean, var_ratio_var


def entropy(y_prob, return_vec=False):
    ent = -1.0 * np.sum(np.multiply(y_prob, np.log(y_prob + np.finfo(float).eps)), axis=1) / np.log(2)
    ent_mean = np.mean(ent)
    ent_var = np.var(ent)
    if return_vec:
        return ent
    return ent_mean, ent_var


def std_score(y_prob, return_vec=False):
    y_mean = np.mean(y_prob, axis=1)[:, np.newaxis]
    std_val = np.sqrt(np.mean(np.square(y_prob - y_mean), axis=1))
    mean_std = np.mean(std_val)
    var_std = np.var(std_val)
    if return_vec:
        return std_val
    return mean_std, var_std


def margin_score(y_prob, return_vec=False):
    y_sorted = np.flip(np.sort(y_prob, axis=1), axis=1)
    ratio = y_sorted[:, 0] - y_sorted[:, 1]
    margin_ratio_mean = np.mean(ratio)
    margin_ratio_var = np.var(ratio)
    if return_vec:
        return ratio
    return margin_ratio_mean, margin_ratio_var


def random(y_prob, return_vec=True):
    uncertainty = np.random.rand(y_prob.shape[0])
    return uncertainty


# Scoring functions

def nll_score(y_prob, y_true, return_vec=False):
    prob_i = np.array([y_prob[j, y_true[j]] for j in range(len(y_prob))])
    nll = -1.0 * np.log(prob_i + np.finfo(float).eps)
    nll_mean = np.mean(nll)
    nll_var = np.var(nll)
    if return_vec:
        return nll
    return nll_mean, nll_var


def avg_spherical_score(y_prob, y_true, return_vec=False):
    # http://faculty.engr.utexas.edu/bickel/Papers/QSL_Comparison.pdf
    # Online Loss: Yes
    prob_i = np.array([y_prob[j, y_true[j]] for j in range(len(y_prob))])
    spherical_loss = 1.0 - (prob_i / (np.sqrt(np.sum(np.square(y_prob))) + np.finfo(float).eps))
    spherical_loss_mean = np.mean(spherical_loss)
    spherical_loss_var = np.var(spherical_loss)
    if return_vec:
        return spherical_loss
    return spherical_loss_mean, spherical_loss_var


def brier_score(y_prob, y_true, num_classes, return_vec=False):
    # https://en.wikipedia.org/wiki/Brier_score, Original definition by Brier
    # Online Loss: Yes
    onehot_y_label = np.zeros((len(y_true), num_classes), dtype=y_prob.dtype)
    onehot_y_label[np.arange(len(y_true)), y_true] = 1

    score = np.mean((y_prob - onehot_y_label) * (y_prob - onehot_y_label), axis=1)
    score_mean = np.mean(score)
    score_var = np.var(score)
    if return_vec:
        return score
    return score_mean, score_var


def ch_score(y_feat, y_true):
    return calinski_harabaz_score(y_feat, y_true)


# Testing helper functions
def label_entropy(lb_encoder, y_true, is_binary=False):
    if is_binary:
        y_sum = 1.0*np.array([y_true.sum(), (1.0-y_true).sum()])
    else:
        y_prob_point = lb_encoder.transform(y_true)*1.0
        y_sum = 1.0*np.sum(y_prob_point, axis=0)
    #print(y_sum)
    y_sum /= np.sum(y_sum)
    entropy = -np.sum(y_sum * np.log(y_sum + np.finfo(float).eps))
    return entropy


if __name__ == "__main__":
    num_classes = 10
    n_points = 10000
    class_vec = np.arange(num_classes)
    y_label = np.random.choice(class_vec, n_points)
    y_pred_raw = np.random.rand(n_points, num_classes)
    y_pred = utils.softmax(y_pred_raw)

    ones = np.ones(n_points)*1.0
    assert(np.allclose(np.sum(y_pred,axis=1),ones))
    enc = preprocessing.LabelBinarizer()
    enc.fit(y_label)

    #Test the functions given in this script

    print("Expected Calibration Error (Calibration, refinement, Brier Multiclass): ",ece(y_pred, y_label, n_bins=15))
    print("F1 Score: ", f1(y_pred, y_label))

    print("Accuracy: ", accuracy(y_pred, y_label))
    print("Entropy (mean, var): ", entropy(y_pred))
    print("Var Ratio (mean, var): ", var_ratio(y_pred))
    print("STD (mean, var): ", std_score(y_pred))
    print("NLL Score (mean, var): ", nll_score(y_pred, y_label))
    print("Average Spherical Score (mean, var): ", avg_spherical_score(y_pred, y_label))
    print("Brier Score (mean,var): ", brier_score(y_pred, y_label, num_classes))
    print("Margin score (mean, var): ", margin_score(y_pred, return_vec=False))
    print("Encval: ", label_entropy(enc, y_label))
