#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  27 10:36:43 2019

@author: Darcane
"""

import torch
from torch import Tensor
from module import Module
import numpy as np


class Sequential(Module):

    def __init__(self, *args):

        self.modules = []
        loss_function = args[-1]


        for idx, arg in enumerate(args):
            self.modules.append(arg)

            if idx < len(args) - 1:
                self.modules[idx].init_loss(loss_function)


    def forward(self, ttt):

        tmp = ttt
        for module in self.modules:
            tmp = module(tmp)
        return tmp

    def backward(self, target):

        tmp = target

        for k in range(len(self.modules)-1, -1, -1):
            tmp = self.modules[k].backward(tmp)

    def calculate_loss(self, target):
        return self.modules[-1].calculate_loss(target)

    def optimize(self, eta):

        for module in self.modules:
            module.optimize(eta)