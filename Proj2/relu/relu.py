#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  27 10:36:43 2019

ReLU activation function

@author: Darcane
"""

import torch
from module import Module


class ReLU(Module):

    def __init__(self):
        self.s = None
        self.x = None
        self.dl_dx = None
        self.dl_ds = None

    @staticmethod
    def activation(s):
        return (s * (s > 0)).type(torch.FloatTensor)

    def forward(self, s):
        # Fill variables for backward pass and apply activation
        self.s = s
        self.x = self.activation(s)
        return self.x

    def backward(self, dl_dx):
        # Compute derivatives
        self.dl_dx = dl_dx
        if self.x is None:
            raise Exception('Forward pass not done yet.')
        self.dl_ds = torch.mul(self.dl_dx, self.dactivation(self.s))
        return self.dl_ds

    def dactivation(self, s):
        # activation'
        return (s > 0).type(torch.FloatTensor)

    def optimize(self, eta):
        # No parameters here, do nothing
        return

    def type(self):
        return 'activation'
