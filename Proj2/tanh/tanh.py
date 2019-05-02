#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  27 10:36:43 2019

@author: Darcane
"""

import torch
from torch import Tensor


class Tanh(Module):

    def __init__(self):
        self.loss = None

    def init_loss(self, loss_function):
        return self.loss = loss_function

    def forward(self, input):
        return (np.exp(2*input) - 1)/(np.exp(2*input) + 1)

    def backward(self):
        raise NotImplementedError

    def dactivation(self, input):
        return 1 - torch.pow(self.activation(input), 2)

    def param(self):
        return []

