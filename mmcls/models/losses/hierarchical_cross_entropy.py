# Copyright (c) OpenMMLab. All rights reserved.
import pdb
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
from .focal_loss import sigmoid_focal_loss


def cross_entropy(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None,
                  masks=None, ):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.size() != label.size():
        label = label.reshape(-1, )
    # element-wise losses
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    if masks is not None:
        loss = loss * masks
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def soft_cross_entropy(pred,
                       label,
                       weight=None,
                       reduction='mean',
                       class_weight=None,
                       avg_factor=None,
                       masks=None, ):
    """Calculate the Soft CrossEntropy loss. The label can be float.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction with shape (N, C).
            When using "mixup", the label can be float.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    if pred.size() != label.size():
        label = label.reshape(-1, )
    loss = -label * F.log_softmax(pred, dim=-1)
    if masks is not None:
        import pdb
        pdb.set_trace()
        loss *= masks
    if class_weight is not None:
        loss *= class_weight
    loss = loss.sum(dim=-1)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         masks=None):
    r"""Calculate the binary CrossEntropy loss with logits.

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        label (torch.Tensor): The gt label with shape (N, \*).
        weight (torch.Tensor, optional): Element-wise weight of loss with shape
            (N, ). Defaults to None.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' , loss
            is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert pred.dim() == label.dim()
    # Ensure that the size of class_weight is consistent with pred and label to
    # avoid automatic boracast,
    label = label.float()
    if class_weight is not None:
        N = pred.size()[0]
        class_weight = class_weight.repeat(N, 1)
    loss = F.binary_cross_entropy_with_logits(
        pred, label, weight=class_weight, reduction='none')
    if masks is not None:
        loss *= masks
    # apply weights and do the reduction
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


@LOSSES.register_module()
class HierarchicalCrossEntropyLoss(nn.Module):

    def __init__(self,
                 levels=2,
                 split=(5, 10),
                 use_sigmoid=(False,),
                 use_soft=(False,),
                 reduction='mean',
                 loss_weight=1.0,
                 use_focal=False,
                 fixed_mask=False,
                 gamma=2.0,
                 alpha=0.5,
                 class_weight=None):
        super(HierarchicalCrossEntropyLoss, self).__init__()
        self.fix_mask = fixed_mask
        self.use_sigmoid = use_sigmoid
        self.use_soft = use_soft
        assert len(split) == levels == len(use_soft) == len(use_sigmoid)
        self.split = split
        for idx, (soft, usigmoid) in \
                enumerate(zip(self.use_soft, self.use_sigmoid)):
            assert not (
                    soft and usigmoid
            ), f'index {idx} in use_sigmoid and use_soft could not be set simultaneously'

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.cls_criterion = list()
        self.label_split = [0]
        for idx in range(levels):
            start = 0 if idx == 0 else self.split[idx - 1]
            end = self.split[idx]
            if self.use_sigmoid[idx]:
                if use_focal:
                    partial_focal = partial(
                        sigmoid_focal_loss, gamma=gamma, alpha=alpha
                    )
                    self.cls_criterion.append(partial_focal)
                else:
                    self.cls_criterion.append(binary_cross_entropy)
                self.label_split.append(self.label_split[-1] + end - start)
            elif self.use_soft[idx]:
                self.cls_criterion.append(soft_cross_entropy)
                self.label_split.append(self.label_split[-1] + 1)
            else:
                self.cls_criterion.append(cross_entropy)
                self.label_split.append(self.label_split[-1] + 1)
        self.label_split = self.label_split[1:]

    def _generate_mask(self, label, cls_score):
        masks = torch.ones_like(cls_score).to(cls_score.device)
        face_label = label[:, 0]
        hand_label = label[:, 1]
        # face, hand | smoke_face, insulating gloves, smoke_hand
        masks[face_label == 1] = torch.tensor([1, 1, 1, 0, 0], dtype=cls_score.dtype).to(cls_score.device)
        masks[hand_label == 1] = torch.tensor([1, 1, 0, 1, 1], dtype=cls_score.dtype).to(cls_score.device)
        return masks

    def _compute_loss(self,
                      cls_score,
                      label,
                      weight,
                      class_weight,
                      reduction,
                      avg_factor,
                      **kwargs
                      ):
        assert max(self.split) == cls_score.size(-1)
        mask = self._generate_mask(label, cls_score) if self.fix_mask else None
        loss_cls = []
        for idx, end in enumerate(self.split):
            start = 0 if idx == 0 else self.split[idx - 1]
            label_start = 0 if idx == 0 else self.label_split[idx - 1]
            label_end = self.label_split[idx]
            # pdb.set_trace()
            cmask = mask[..., label_start:label_end] if mask is not None else mask
            single_loss = self.cls_criterion[idx](cls_score[..., start:end],
                                                  label[..., label_start:label_end],
                                                  weight,
                                                  class_weight=class_weight,
                                                  reduction=reduction,
                                                  avg_factor=avg_factor,
                                                  masks=cmask,
                                                  **kwargs
                                                  )
            loss_cls.append(single_loss)

        return torch.stack(loss_cls).mean()

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        # pdb.set_trace()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        loss_cls = self.loss_weight * self._compute_loss(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        # pdb.set_trace()
        return loss_cls
