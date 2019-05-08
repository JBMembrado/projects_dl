#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  27 10:36:43 2019

@author: Darcane
"""

import torch
from torch import Tensor
from linear import Linear
from tanh import Tanh
from sequential import Sequential
from mse import MSE
import numpy as np


def generate_fake_samples(N = 1000):
    # N being the number of training and test samples we want to generate
    train_samples = 2*torch.rand(N, 2) - torch.ones(N, 2)
    test_samples = 2*torch.rand(N, 2) - torch.ones(N, 2)

    train_labels = torch.ones(N)
    test_labels = torch.ones(N)

    for k in range(N):
        if train_samples[k, 0]**2 + train_samples[k, 1]**2 > 1/(2*np.pi):
            train_labels[k] = 0
        if test_samples[k, 0]**2 + test_samples[k, 1]**2 > 1:
            test_labels[k] = 0

    return train_samples, test_samples, train_labels, test_labels


test = Sequential(Linear(3, 5), Tanh(), Linear(5, 1), MSE())
x_input = Tensor([1, -1, 2,2,3,4]).view(-1, 3)
target = Tensor([[10.0], [1.0]])

# Training the Net

eta = 0.01


# for i in range(50):
#     test.forward(x_input)
#     print(test.calculate_loss(target))
#     test.backward(target)
#     test.optimize(eta)

a, b, c, d = generate_fake_samples(5)

print(a)
print(c)