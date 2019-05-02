#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  27 10:36:43 2019

@author: Darcane
"""

import torch
from torch import Tensor


class Tanh(Module):

    def __init__(self):
        self.loss = None

    def forward(self, *input):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def param(self):
        return []

