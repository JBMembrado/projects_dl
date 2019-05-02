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
from activation_functions import *
import numpy as np

class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = Linear(3, 5)
        self.fc2 = Linear(5, 1)

        self.act1 = Tanh()

    def forward(self, x):
        x = Functions.tanh(self.fc1(x.view(-1, 3)))
        x = self.fc2(x)
        return x

#    def backward(self, target):
#        return


test = Net()
x_input = Tensor([1, -1, 2])
print(test.forward(x_input))

nn.Sequential