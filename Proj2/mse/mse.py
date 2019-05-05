#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  27 10:36:43 2019

@author: Darcane
"""

import torch
from module import Module
import numpy as np
from torch import Tensor


class MSE(Module):

    def __init__(self):
        return

    def calculate_loss(self, target):
        return (self.output - target).item()**2

    def derivate_loss(self, target):
        return 2*(self.output - target).item()

    def forward(self, input):
        self.output = input
        return self.output

    def type(self):
        return 'loss'

    def param(self):
        return []

    def __call__(self, input):
        return self.forward(input)