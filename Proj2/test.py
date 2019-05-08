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



test = Sequential(Linear(3, 5), Tanh(), Linear(5, 1), MSE())
x_input = Tensor([1, -1, 2,2,3,4]).view(-1, 3)
target = Tensor([[10.0], [1.0]])

# Training the Net

eta = 0.01


for i in range(50):
    test.forward(x_input)
    print(test.calculate_loss(target))
    test.backward(target)
    test.optimize(eta)
