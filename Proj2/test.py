#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  27 10:36:43 2019

@author: Darcane
"""

import torch
from torch import nn
from torch import Tensor
from module import Module
from linear import Linear
from tanh import Tanh
from sequential import Sequential
from mse import MSE
from activation_functions import *
import numpy as np

class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lay1 = Linear(3, 5)
        self.lay2 = Linear(5, 1)

        self.act1 = Tanh()

        self.loss_function = MSE()

        self.sequence = Sequential(self.lay1, self.act1, self.lay2, self.loss_function)

    def forward(self, x):
        return self.sequence.forward(x.view(-1, 3))

#    def backward(self, target):
#        return


test = Net()
x_input = Tensor([1, -1, 2])
target = Tensor([[10.0]])
test.forward(x_input)

# print(test.lay1.s)
# print(test.act1.s)
# print(test.act1.x)
# print(test.lay2.x)
# print(test.lay2.s)

print(test.lay2.s)
print(target)
print(test.loss_function.calculate_loss(target))