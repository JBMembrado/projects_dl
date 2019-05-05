#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  27 10:36:43 2019

@author: Darcane
"""

import torch
from module import Module
import numpy as np
from torch import Tensor


class Tanh(Module):

    def __init__(self):
        self.loss = None

    def init_loss(self, loss_function):
        self.loss = loss_function

    def forward(self, input):
        self.s = input
        self.x = (np.exp(2*input) - 1)/(np.exp(2*input) + 1)
        return self.x

    def backward(self):
        raise NotImplementedError

    def dactivation(self, input):
        return 1 - torch.pow(self.activation(input), 2)

    def type(self):
        return 'activation'

    def param(self):
        return []

    def __call__(self, input):
        return self.forward(input)
