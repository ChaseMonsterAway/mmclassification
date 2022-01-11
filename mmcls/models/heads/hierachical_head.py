# Copyright Jun Sun.
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cls_head import ClsHead
from ..builder import HEADS


@HEADS.register_module()
class HiearachicalLinearClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 file_type=None,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(HiearachicalLinearClsHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)
        assert file_type is not None
        self.file_type = file_type
        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def _act(self, cls_score):
        # return cls_score
        new_cls_score = []
        """[dict(type='ce', max_len=3), dict(type='bce', max_len=2)]"""
        split_len = [ftype['max_len'] for ftype in self.file_type]
        split_len.insert(0, 0)
        split_len = np.cumsum(split_len).tolist()
        act_type = ['softmax' if ftype['type'] == 'ce' else 'sigmoid'
                    for ftype in self.file_type]
        for idx, start in enumerate(split_len[:-1]):
            end = split_len[idx + 1]
            act_func = partial(F.softmax, dim=-1) \
                if act_type[idx] == 'softmax' else torch.sigmoid
            # cls_score[..., start:end] = act_func(cls_score[..., start:end])
            if act_type[idx] != 'softmax':
                for i in range(start, end):
                    new_cls_score.append(act_func(cls_score[..., i:i + 1]))
            else:
                new_cls_score.append(act_func(cls_score[..., start:end]))

        # return cls_score
        # pdb.set_trace()
        return torch.cat(new_cls_score, dim=1)

    def simple_test(self, x):
        """Test without augmentation."""
        if isinstance(x, tuple):
            x = x[-1]
        cls_score = self.fc(x)
        if isinstance(cls_score, list):
            raise "Unsupported type list"
            # cls_score = sum(cls_score) / float(len(cls_score))
        # pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        pred = self._act(cls_score)

        return self.post_process(pred)

    def _compute_ignore_masks(self, gt_label):
        mask = torch.ones_like(gt_label)
        mask[gt_label == -1] = 0
        return mask

    def loss(self, cls_score, gt_label, masks):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        loss = self.compute_loss(cls_score, gt_label, avg_factor=num_samples, masks=masks)
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses['accuracy'] = {
                f'top-{k}': a
                for k, a in zip(self.topk, acc)
            }
        losses['loss'] = loss
        return losses

    def forward_train(self, x, gt_label):
        if isinstance(x, tuple):
            x = x[-1]
        cls_score = self.fc(x)
        masks = self._compute_ignore_masks(gt_label)
        losses = self.loss(cls_score, gt_label, masks=masks)
        return losses
