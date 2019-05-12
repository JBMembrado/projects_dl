#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  27 10:36:43 2019

@author: Darcane
"""

import torch
from linear import Linear
from tanh import Tanh
from relu import ReLU
from sequential import Sequential
from mse import MSE
import numpy as np


def generate_fake_samples(n=1000):
    # N being the number of training and test samples we want to generate
    train_samples = torch.rand(n, 2)
    test_samples = torch.rand(n, 2)
    train_labels = torch.rand(n, 2)
    test_labels = torch.rand(n, 2)

    train_labels[:, 0] = (train_samples[:, 0] - 0.5)**2 + (train_samples[:, 1] - 0.5)**2 < 1/(2*np.pi)
    train_labels[:, 1] = 1 - train_labels[:, 0]
    test_labels[:, 0] = (test_samples[:, 0] - 0.5)**2 + (test_samples[:, 1] - 0.5)**2 < 1/(2*np.pi)
    test_labels[:, 1] = 1 - test_labels[:, 0]
    return train_samples, test_samples, train_labels.type(torch.FloatTensor), test_labels.type(torch.FloatTensor)


test = Sequential(Linear(2, 25), Tanh(), Linear(25, 25), Tanh(), Linear(25, 25), Tanh(), Linear(25, 2), MSE())

# Training the Net

# Generating the train and test data
train_x, test_x, train_target, test_target = generate_fake_samples(1000)

n_samples, dim_input = train_x.size()

# Setting the number of gradient steps we want to do and the value of eta
gradient_steps = 1000
eta = 0.01/n_samples

# Gradient descent to train on the training data
for step in range(gradient_steps):
    test(train_x)
    if step % 100 == 0:
        print('For step', step, 'we have the loss', test.calculate_loss(train_target).item())
        # print('dl_dw for layer 1', test.modules[0].dl_dx)

    test.backward(train_target)
    test.optimize(eta)

# Once it is trained, we can try on the test data

# Prediction with the trained network
pred_target = test(train_x)
print('Train error :', torch.sum(pred_target.argmax(1) != train_target.argmax(1)).item()/n_samples*100, '%')

pred_target = test(test_x)
print('Test error :', torch.sum(pred_target.argmax(1) != test_target.argmax(1)).item()/n_samples*100, '%')
