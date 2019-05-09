#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  27 10:36:43 2019

@author: Darcane
"""

import torch
from module import Module


class MSE(Module):

    def __init__(self):
        self.output = None

    def calculate_loss(self, target):
        return torch.mean((self.output - target)**2)

    def derivate_loss(self, target):
        # print('size of derivate loss', target)
        return 2*(self.output - target)

    def forward(self, output):
        self.output = output
        return self.output

    def backward(self, target):
        if self.output is None:
            raise Exception('Forward pass not done yet.')
        return self.derivate_loss(target)

    def type(self):
        return 'loss'

    def param(self):
        return []

    def __call__(self, x):
        return self.forward(x)