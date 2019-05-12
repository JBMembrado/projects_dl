#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  27 10:36:43 2019

@author: Darcane
"""

from module import Module


class Sequential(Module):

    def __init__(self, *args):

        self.modules = []

        if len(args) < 1:
            raise Exception("No module given when initialising class Sequential.")

        if not hasattr(args[-1], 'type'):
            raise Exception("The last argument has no method 'type'.")

        if args[-1].type() != 'loss':
            raise Exception("The last argument is not a loss function. Please set the loss function to be used as \
                the last argument of the Sequential object")

        for idx, arg in enumerate(args):
            if not isinstance(arg, Module):
                raise Exception("Arg given in position {} is not an instance of a subclass of Module.".format(idx))

            self.modules.append(arg)

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