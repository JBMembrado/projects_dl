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


class Functions(Module):

    def __init__(self):
        pass

    def ReLU(input):
        return torch.max(0, input)

    def sigmoid(input):
        return 1/(1 + np.exp(-input))

    def tanh(input):
        return (np.exp(2*input) - 1)/(np.exp(2*input) + 1)

    def dtanh(input):
        # The derivate of the tanh function defined above
        return 1 - torch.pow(tanh(inputs), 2)

    def loss(verit, target):
        return (torch.sum(torch.pow(target - verit, 2)))

    def dloss(verit, target):
        return 2*(verit - target)

    def linear(input, weight, bias):
        "Linear transformation of the input data according to : y = xA^T + b"
        output = input.mm(weight.t()) + bias
        return output

