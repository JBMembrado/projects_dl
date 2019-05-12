#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  27 10:36:43 2019

@author: Darcane
"""

import torch
from torch import Tensor
from module import Module


class Linear(Module):

    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(out_features, in_features)

        self.bias = Tensor(out_features)
        self.epsilon = 1

        self.x = None
        self.s = None
        self.dl_dx = None
        self.dl_ds = None

        self.dl_dw = Tensor(out_features, in_features)
        self.dl_db = Tensor(out_features)

        self.init_parameters()

    def init_parameters(self):
        self.weight = torch.randn(self.out_features, self.in_features)*self.epsilon
        self.bias = torch.randn(self.out_features)*self.epsilon

    def forward(self, x):
        self.x = x
        # print('x ', x.shape)
        # print('weight ', self.weight.shape)
        self.s = torch.mm(x, self.weight.t()) + self.bias
        return self.s

    def backward(self, dl_ds):
        self.dl_ds = dl_ds
        # print('dl_ds ', dl_ds.shape)
        # print('weight ', self.weight.shape)
        self.dl_dx = torch.mm(self.dl_ds, self.weight)

        self.dl_dw = torch.mm(self.dl_ds.t(), self.x)
        self.dl_db = self.dl_ds.sum(0)

        return self.dl_dx

    def optimize(self, eta):
        self.weight = self.weight - eta * self.dl_dw
        self.bias = self.bias - eta * self.dl_db
        return

    def type(self):
        return 'layer'

