# -*- coding: utf-8 -*-
# @Time    : 2022/10/24 15:33
# @Author  : sylviazz
# @FileName: layers
import torch.nn as nn
import torch
import numpy as np
class GatedDilatedResidualConv1D(nn.Module):
    def __init__(self, dim: int, dilation_rate: int):
        super(GatedDilatedResidualConv1D, self).__init__()
        self.dim = dim
        self.conv1d = Conv1D(input_dim=self.dim, output_dim=2 * self.dim, kernel_size=3, dilation_rate=dilation_rate)
        self.dropout = nn.Dropout(0.1)
    def forward(self, seq, mask):
        c = self.conv1d(seq)

        def _gate(x):
            dropout_rate = 0.1
            s, h = x
            g, h = h[:, :, :self.dim], h[:, :, self.dim:]
            g = self.dropout(g)
            g = torch.sigmoid(g)
            return g * s + (1 - g) * h

        seq = _gate([seq, c])
        seq = seq * mask
        return seq
class Conv1D(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, act: str = None, dilation_rate: int = 1):
        super(Conv1D, self).__init__()
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.rec_field = self.kernel_size + (self.kernel_size - 1) * (self.dilation_rate - 1)
        self.pad = self.rec_field // 2
        self.conv1d = nn.Conv2d(in_channels=1, out_channels=output_dim, kernel_size=(3, input_dim),
                                           padding=(self.pad, 0), dilation=(dilation_rate, 1))

    def forward(self, seq):
        h = seq.unsqueeze(1).cuda()
        h = h.type(torch.FloatTensor).cuda()
        h = self.conv1d(h)
        h = h.squeeze(-1)
        h = h.transpose(2, 1)
        return h