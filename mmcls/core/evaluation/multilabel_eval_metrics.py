# Copyright (c) OpenMMLab. All rights reserved.
import pdb
import warnings

import numpy as np
import torch

from sklearn.metrics import roc_curve, auc

def tpr_with_fix_fpr(pred, gt, value):
    fpr, tpr, thr = roc_curve(gt, pred)
    ind = np.where(fpr<=value)[0][-1]
    auc_area = auc(fpr, tpr)
    return fpr[ind], tpr[ind], thr[ind], auc_area


def average_performance(pred, target, thr=None, k=None, class_wise=False,
        value=0.05):
    """Calculate CP, CR, CF1, OP, OR, OF1, where C stands for per-class
    average, O stands for overall average, P stands for precision, R stands for
    recall and F1 stands for F1-score.

    Args:
        pred (torch.Tensor | np.ndarray): The model prediction with shape
            (N, C), where C is the number of classes.
        target (torch.Tensor | np.ndarray): The target of each prediction with
            shape (N, C), where C is the number of classes. 1 stands for
            positive examples, 0 stands for negative examples and -1 stands for
            difficult examples.
        thr (float): The confidence threshold. Defaults to None.
        k (int): Top-k performance. Note that if thr and k are both given, k
            will be ignored. Defaults to None.
        class_wise (bool): If True, will return precision and recall of each class.

    Returns:
        tuple: (CP, CR, CF1, OP, OR, OF1)
    """
    nums=1 if pred.ndim == 1 else pred.shape[-1]
    each_class_info = list()
    for num in nums:
        p = pred[..., num]
        g = target[..., num]
        fpr, tpr, thr, auc_area = tpr_with_fix_fpr(p, g, value=value)
        each_class_info.append(np.array(np.round(fpr, 4), np.round(tpr, 4),
            np.round(thr, 4), np.round(auc_area, 4), '||'))
    tpr_at_fpr = np.concatenate(each_class_info, axis=0)

    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or'
                        'np.ndarray')
    if thr is None and k is None:
        thr = 0.5
        warnings.warn('Neither thr nor k is given, set thr as 0.5 by '
                      'default.')
    elif thr is not None and k is not None:
        warnings.warn('Both thr and k are given, use threshold in favor of '
                      'top-k.')

    assert pred.shape == \
           target.shape, 'pred and target should be in the same shape.'

    eps = np.finfo(np.float32).eps
    target[target == -1] = 0
    if thr is not None:
        # a label is predicted positive if the confidence is no lower than thr
        pos_inds = pred >= thr

    else:
        # top-k labels will be predicted positive for any example
        sort_inds = np.argsort(-pred, axis=1)
        sort_inds_ = sort_inds[:, :k]
        inds = np.indices(sort_inds_.shape)
        pos_inds = np.zeros_like(pred)
        pos_inds[inds[0], sort_inds_] = 1

    tp = (pos_inds * target) == 1
    fp = (pos_inds * (1 - target)) == 1
    fn = ((1 - pos_inds) * target) == 1
    tn = np.logical_and(1 - pos_inds, target == 0)
    acc_class = (tp.sum(axis=0) + tn.sum(axis=0)) / (tp.sum(axis=0) + \
                tn.sum(axis=0) + fn.sum(axis=0) + fp.sum(axis=0))
    precision_class = tp.sum(axis=0) / np.maximum(
        tp.sum(axis=0) + fp.sum(axis=0), eps)
    recall_class = tp.sum(axis=0) / np.maximum(
        tp.sum(axis=0) + fn.sum(axis=0), eps)
    tpr = tp.sum(axis=0) / np.maximum(tp.sum(axis=0) + fn.sum(axis=0), eps) * 100
    fpr = fp.sum(axis=0) / np.maximum(fp.sum(axis=0) + tn.sum(axis=0), eps) * 100
    CP = precision_class.mean() * 100.0
    CR = recall_class.mean() * 100.0
    CF1 = 2 * CP * CR / np.maximum(CP + CR, eps)
    OP = tp.sum() / np.maximum(tp.sum() + fp.sum(), eps) * 100.0
    OR = tp.sum() / np.maximum(tp.sum() + fn.sum(), eps) * 100.0
    OF1 = 2 * OP * OR / np.maximum(OP + OR, eps)
    if class_wise:
        C_wise_F1 = 100 * 2 * precision_class * recall_class / \
                    np.maximum(precision_class + recall_class, eps)
        C_wise_F1 = np.concatenate([np.round(C_wise_F1, 2), np.round(tpr, 2),
            np.round(fpr, 2), tpr_at_fpr])
        return CP, CR, CF1, C_wise_F1, np.round(100 * acc_class, 2), OP, OR, OF1

    return CP, CR, CF1, OP, OR, OF1
