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
from math import pi

# Disable autograd
torch.set_grad_enabled(False)


def generate_fake_samples(n=1000):
    """
    Generate points in a 1 by 1 square and identify those within the circle of radius 1/sqrt(2 pi) (label 1)
    and the others (label 0)

    :param n: number of samples to generate
    :return: train_samples, test_samples, train_labels, test_labels
    """
    train_samples = torch.rand(n, 2)#(n,2) tensor of values between 0 and 1
    test_samples = torch.rand(n, 2)
    train_labels = torch.rand(n, 2)
    test_labels = torch.rand(n, 2)

    train_labels[:, 0] = (train_samples[:, 0] - 0.5) ** 2 + (train_samples[:, 1] - 0.5) ** 2 < 1 / (2 * pi)
    train_labels[:, 1] = 1 - train_labels[:, 0]
    test_labels[:, 0] = (test_samples[:, 0] - 0.5) ** 2 + (test_samples[:, 1] - 0.5) ** 2 < 1 / (2 * pi)
    test_labels[:, 1] = 1 - test_labels[:, 0]
    return train_samples, test_samples, train_labels.type(torch.FloatTensor), test_labels.type(torch.FloatTensor)


# Generating the train and test data
train_x, test_x, train_target, test_target = generate_fake_samples(1000)
n_samples, dim_input = train_x.size()

# Define Net
test = Sequential(Linear(2, 25), Tanh(), Linear(25, 25), Tanh(), Linear(25, 25), Tanh(), Linear(25, 2), MSE())

# Setting the number of gradient steps we want to do and the value of eta
gradient_steps = 1000
eta = 0.01 / n_samples

# Gradient descent to train on the training data
for step in range(gradient_steps):
    test(train_x)
    if step % 100 == 0:#(just to have less things displayed)
        print('For step', step, 'we have the loss', test.calculate_loss(train_target).item())
    test.backward(train_target)
    test.optimize(eta)


# Predictions with the trained network
pred_target = test(train_x)
print('Train error :', torch.sum(pred_target.argmax(1) != train_target.argmax(1)).item() / n_samples * 100, '%')

pred_target = test(test_x)
print('Test error :', torch.sum(pred_target.argmax(1) != test_target.argmax(1)).item() / n_samples * 100, '%')
