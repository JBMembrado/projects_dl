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
        for arg in args:
            self.modules.append(arg)

        return

    def forward(self, input):

        tmp = input
        for module in self.modules:
            tmp = module(tmp)
        return tmp

    def backward(self):

