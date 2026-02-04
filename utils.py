import os
import sys
import datetime
import random

import torch
import numpy as np

from scipy.optimize import linear_sum_assignment


def project_simplex(B):

    n, m = B.shape
    A = np.tile(np.arange(1, m+1), (n, 1))

    B_sort = np.sort(B, axis=1)[:, ::-1]  # 降序排序
    cum_B = np.cumsum(B_sort, axis=1)

    sigma = B_sort - (cum_B - 1) / A

    tmp = sigma > 0
    idx = np.sum(tmp, axis=1)

    tmp = B_sort - sigma
    sigma = np.diag(tmp[np.arange(n), idx - 1])  # 注意索引从0开始，所以需要减1
    # sigma = np.tile(sigma, (1, m))

    X = np.maximum(B - sigma, 0)

    return X, sigma

def forward_prob(q_i, q_j):

    p_i = q_i.sum(0).view(-1)
    p_i /= p_i.sum()
    ne_i = (p_i * torch.log(p_i)).sum()

    p_j = q_j.sum(0).view(-1)
    p_j /= p_j.sum()
    ne_j = (p_j * torch.log(p_j)).sum()

    entropy = ne_i + ne_j

    return entropy

def create_dir_model(model_parent_dir):
    record_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S')
    model_dir = model_parent_dir + record_time
    if not os.path.exists('./models'):
        os.mkdir('./models')
    if not os.path.exists(model_parent_dir):
        os.mkdir(model_parent_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    return model_dir, record_time

def labels_to_adjacency(labels):
    labels = labels.view(-1, 1)
    adj = (labels == labels.T).int()

    return adj


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def get_cluster_acc(y_pred, y_true):
    """
    Calculate clustering accuracy and clustering mean per class accuracy.
    Requires scipy installed
    # Arguments
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        y_true: true labels, numpy.array with shape `(n_samples,)`
    # Return
        Accuracy in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    match = np.array(list(map(lambda i: col_ind[i], y_pred)))

    mean_per_class = [0 for i in range(D)]
    for c in range(D):
        mask = y_true == c
        mean_per_class[c] = np.mean((match[mask] == y_true[mask]))
    # mean_per_class_acc = np.mean(mean_per_class)

    correct_num = w[row_ind, col_ind].sum()

    return correct_num / y_pred.size, correct_num

