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

    train_labels = torch.ones(N, 1)
    test_labels = torch.ones(N, 1)

    for k in range(N):
        if train_samples[k, 0]**2 + train_samples[k, 1]**2 > 1/(2*np.pi):
            train_labels[k] = 0
        if test_samples[k, 0]**2 + test_samples[k, 1]**2 > 1:
            test_labels[k] = 0

    return train_samples, test_samples, train_labels, test_labels


test = Sequential(Linear(2, 10), Tanh(), Linear(10, 5), Tanh(), Linear(5, 1), MSE())

# Training the Net

# Generating the train and test data
train_samples, test_samples, train_target, test_target = generate_fake_samples(1000)

target_coeff = 0.9
train_target = train_target*target_coeff
test_target = test_target*target_coeff

n_samples, dim_input = train_samples.size()

# Setting the number of gradient steps we want to do and the value of eta
gradient_steps = 1000
eta = 0.1

# Gradient descent to train on the training data
for step in range(gradient_steps):

    test.forward(train_samples)
    if step % 100 == 0:
        print('For step', step, 'we have the loss', test.calculate_loss(train_target).item())
        # print('dl_dw for layer 1', test.modules[0].dl_dx)

    test.backward(train_target)
    test.optimize(eta)

# Once it is trained, we can try on the test data

# Prediction with the trained network
pred_target = test.forward(test_samples)

pred_target = np.round(pred_target)
test_target = np.round(test_target)

# print(pred_target)

# Number of errors by comparing the prediction and the true target labels
print('Number of errors :', torch.sum(pred_target != test_target).item())


