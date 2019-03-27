#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:36:43 2019

@author: Darcane
"""

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

import dlc_practical_prologue as prologue


# 2 Layers network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 2)

        self.mini_batch_size = 100
        self.eta = 1e-3
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=5, stride=5))
        x = F.relu(self.fc1(x.view(-1, 128)))
        x = self.fc2(x)
        return x

    # Training Function
    def train(self, train_input, train_target):
        for e in range(100):
            sum_loss = 0
            for b in range(0, train_input.size(0), self.mini_batch_size):
                output = self(train_input.narrow(0, b, self.mini_batch_size))
                loss = self.criterion(output, train_target.narrow(0, b, self.mini_batch_size))
                self.zero_grad()
                loss.backward()
                sum_loss = sum_loss + loss.item()
                for p in self.parameters():
                    p.data.sub_(self.eta * p.grad.data)
            print("Step %d : %f" % (e, sum_loss))

    # Test error
    def nb_errors(self, input_data, target):
        nb_errors = 0
        for b in range(0, input_data.size(0), self.mini_batch_size):
            output = self(input_data.narrow(0, b, self.mini_batch_size))
            predictions = output.argmax(1)
            target_labels = target.narrow(0, b, self.mini_batch_size)
            nb_errors += torch.sum(target_labels != predictions)
        return float(nb_errors) * 100 / input_data.size(0)


""" Load Data """
N_PAIRS = 1000

train_input, train_target, train_classes, test_input, test_target, test_classes = \
    prologue.generate_pair_sets(N_PAIRS)

N = train_input.size(0)

train_input, train_target = Variable(train_input), Variable(train_target)
test_input, test_target = Variable(test_input), Variable(test_target)

""" Create and train model """
my_model = Net()

my_model.train(train_input, train_target)
print("Train error : %.1f%% \nTest error : %.1f%%" %
      (my_model.nb_errors(train_input, train_target),
       my_model.nb_errors(test_input, test_target)))
