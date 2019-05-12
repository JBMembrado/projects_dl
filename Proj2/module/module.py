#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  27 10:36:43 2019

@author: Darcane
"""


class Module(object):

    def __init__(self):
        self.loss = None

    def forward(self):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

    def backward(self):
        raise NotImplementedError

    def optimize(self, eta):
        return

    def type(self):
        raise NotImplementedError

    def param(self):
        return []

