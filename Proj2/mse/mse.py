#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  27 10:36:43 2019

MSE loss

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
        # loss'
        return 2*(self.output - target)

    def forward(self, output):
        # Fill variables for backward pass and apply activation
        self.output = output
        return self.output

    def backward(self, target):
        # Compute derivatives
        if self.output is None:
            raise Exception('Forward pass not done yet.')
        return self.derivate_loss(target)

    def optimize(self, eta):
        # No parameters here, do nothing
        return

    def type(self):
        return 'loss'
