#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  27 10:36:43 2019

@author: Darcane
"""

import torch
from torch import Tensor
from module import Module
from activation_functions import *
import numpy as np



class Linear(Module):

    def __init__(self, in_features, out_features, next_layer):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(out_features, in_features)

        self.bias = Tensor(out_features)
        self.x = Tensor(in_features)
        self.epsilon = 1e1
        self.init_parameters()

        if nargin <=2:
            self.next_layer = None
        else:
            self.next_layer = next_layer

    def init_parameters(self):
        self.weight = torch.randn(self.out_features, self.in_features)*self.epsilon
        self.bias = torch.randn(self.out_features)*self.epsilon
        self.x = torch.randn(self.in_features)*self.epsilon

    def forward(self, input):
        return input.mm(self.weight.t()) + self.bias

    def backward(self, target):
        dl_dx = Functions.dtanh(self.x, target)
        return dl_dx

    def init_loss

    def __call__(self, input):
        return self.forward(input)

