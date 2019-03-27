#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:36:43 2019

@author: Darcane
"""

from torch.autograd import Variable
import dlc_practical_prologue as prologue
from model import Net


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
