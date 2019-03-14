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

n_pairs = 1000

train_input, train_target, train_classes, test_input, test_target, test_classes = \
prologue.generate_pair_sets(n_pairs)

N = train_input.size(0)

train_target = torch.tensor([ [1,0] if train_target[i]==0 else [0,1] for i in range(N)])
test_target = torch.tensor([ [1,0] if test_target[i]==0 else [0,1] for i in range(N)])



# 2 Layers network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3)
        self.fc1 = nn.Linear(128, 200)
        self.fc2 = nn.Linear(200, 2)
 
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=3, stride=3))
        x = F.relu(self.fc1(x.view(-1, 128)))
        x = self.fc2(x)
        return x


train_input, train_target = Variable(train_input), Variable(train_target)

model, criterion = Net(), nn.MSELoss()
eta, mini_batch_size = 1e-1, 100

# Training Function

def train_model(model, train_input, train_target, test_input, test_target,  mini_batch_size):
    
    for e in range(0, 25):
        sum_loss = 0
        # We do this with mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
  
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.item()
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.data.sub_(eta * p.grad.data)
        print(e, sum_loss)
        print("Percentage of errors  on the train = ", compute_nb_errors(model, train_input, train_target, mini_batch_size))
        print("Percentage of errors  on the test = ", compute_nb_errors(model, test_input, test_target, mini_batch_size))

        
    return (output, sum_loss)
    

# Test error

def compute_nb_errors(model, input_data, target, mini_batch_size):
    
    nb_errors = 0
    for b in range(0, input_data.size(0), mini_batch_size):

        output = model(input_data.narrow(0, b, mini_batch_size))        
        predictions = torch.round(output)
        target_labels = target.narrow(0, b, mini_batch_size)

        nb_errors += torch.sum(predictions.long() != target_labels.long()).item()
    
    return nb_errors*100/(input_data.size(0))
        

    






    

