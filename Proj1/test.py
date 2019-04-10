#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:36:43 2019

@author: Darcane
"""

from torch.autograd import Variable
import dlc_practical_prologue as prologue
from my_nets import *


""" Load Data """
N_PAIRS = 1000

train_input, train_target, train_classes, test_input, test_target, test_classes = \
    prologue.generate_pair_sets(N_PAIRS)

train_input, train_target = Variable(train_input), Variable(train_target)
test_input, test_target = Variable(test_input), Variable(test_target)

""" Create and train model """
my_model = NetWithBatchNorm()

my_model.trainer(train_input, train_target)
print("Train error : %.1f%% \nTest error : %.1f%%" %
      (my_model.nb_errors(train_input, train_target),
       my_model.nb_errors(test_input, test_target)))

""" Create and train model which identifies each number and then compares them """
my_model_number = NetNumber()
my_model_number.trainer_nb(train_input, train_classes)
print("Train error : %.1f%% \nTest error : %.1f%%" %
      (my_model_number.nb_errors_nb(train_input, train_target),
       my_model_number.nb_errors_nb(test_input, test_target)))
