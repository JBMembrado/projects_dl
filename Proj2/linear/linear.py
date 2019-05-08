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
        self.epsilon = 1e-1

        self.x = Tensor(in_features)
        self.s = Tensor(out_features)

        self.dl_dx = Tensor(in_features)
        self.dl_ds = Tensor(out_features)
        self.dl_dw = Tensor(out_features, in_features)
        self.dl_db = Tensor(out_features)

        self.init_parameters()

    def init_parameters(self):
        self.weight = torch.randn(self.out_features, self.in_features)*self.epsilon
        self.bias = torch.randn(self.out_features)*self.epsilon
        self.x = torch.randn(self.in_features)*self.epsilon

    def forward(self, x):
        self.x = x
        self.s = torch.mm(x, self.weight.t()) + self.bias
        return self.s

    def backward(self, dl_ds):
        self.dl_ds = dl_ds
        self.dl_dx = torch.mm(self.dl_ds, self.weight)

        self.dl_dw = torch.mm(self.dl_ds.t(), self.x)
        self.dl_db = self.dl_ds.mean(0)

        return self.dl_dx

    def optimize(self, eta):
        self.weight = self.weight - eta * self.dl_dw
        self.bias = self.bias - eta * self.dl_db
        return

    def type(self):
        return 'layer'

    def __call__(self, x):
        return self.forward(x)

