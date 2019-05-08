#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  27 10:36:43 2019

@author: Darcane
"""

import torch
from module import Module
import numpy as np


class Tanh(Module):

    def __init__(self):
        self.s = None
        self.x = None

        self.dl_dx = None
        self.dl_ds = None

    def activation(self, s):
        return np.tanh(s)

    def forward(self, s):
        self.s = s
        self.x = self.activation(s)
        return self.x

    def backward(self, dl_dx):
        self.dl_dx = dl_dx
        if self.x is None:
            raise Exception('Forward pass not done yet.')
        self.dl_ds = torch.mul(self.dl_dx, self.dactivation(self.x))

        return self.dl_ds

    def dactivation(self, s):
        return 1 - torch.pow(self.activation(s), 2)

    def type(self):
        return 'activation'

    def param(self):
        return []

    def __call__(self, s):
        return self.forward(s)
