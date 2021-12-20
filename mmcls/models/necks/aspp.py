# modify from https://github.com/pytorch/vision/tree/master/torchvision/models/segmentation/deeplabv3.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from mmcv.runner import BaseModule

from ..builder import NECKS


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, norm_layer,
                 act_layer):
        modules = [
            nn.Conv2d(in_channels,
                      out_channels,
                      3,
                      padding=dilation,
                      dilation=dilation,
                      bias=False),
            norm_layer(out_channels),
            act_layer()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_layer, act_layer,
                 mode='nearest', align_corners=False):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels), act_layer())
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        y = super(ASPPPooling, self).forward(x)
        return F.interpolate(y,
                             size=(int(x.size(2)), int(x.size(3))),
                             mode=self.mode,
                             align_corners=self.align_corners)


@NECKS.register_module()
class ASPP(BaseModule):
    def __init__(self, in_channels, out_channels, atrous_rates,
                 mode='nearest', align_corners=False, dropout=None,
                 init_cfg=None,
                 ):
        super(ASPP, self).__init__(init_cfg=init_cfg)
        norm_layer = nn.BatchNorm2d
        act_layer = partial(nn.ReLU, inplace=True)

        modules = [nn.Sequential(nn.Conv2d(in_channels,
                                           out_channels,
                                           1,
                                           bias=False),
                                 norm_layer(out_channels),
                                 act_layer())]

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(
            ASPPConv(in_channels, out_channels, rate1, norm_layer, act_layer))
        modules.append(
            ASPPConv(in_channels, out_channels, rate2, norm_layer, act_layer))
        modules.append(
            ASPPConv(in_channels, out_channels, rate3, norm_layer, act_layer))
        modules.append(
            ASPPPooling(in_channels, out_channels, norm_layer, act_layer,
                        mode=mode, align_corners=align_corners))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels), act_layer())
        self.with_dropout = dropout is not None
        if self.with_dropout:
            self.dropout = nn.Dropout(dropout)

    def single_forward(self, x):
        res = []
        for conv in self.convs:
            x = conv(x)
            res.append(x)
        res = torch.cat(res, dim=1)
        res = self.project(res)
        if self.with_dropout:
            res = self.dropout(res)
        return res

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.single_forward(x) for x in inputs])
        elif isinstance(inputs, torch.Tensor):
            outs = self.single_forward(inputs)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')

        return outs
