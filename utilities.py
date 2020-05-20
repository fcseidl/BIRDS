#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 10:03:22 2020

@author: fcseidl

Various helper classes and functions for Koopman mode estimation.
"""

import numpy as np
import sklearn.gaussian_process as gp

def kron_delta(i, j):
    r"""
    Kronecker delta.
    """
    if i == j:
        return 1
    return 0

## TODO: is this any different from WhiteKernel?
class KronKernel(gp.kernels.StationaryKernelMixin, gp.kernels.Kernel):
    r"""
    Returns cov(x, x') = Kronecker delta of x and x'. Overrides an abstract 
    kernel class from sklearn.
    """
    
    def __init__(self):
        # no hyperparameters
        pass
    
    def __call__(self, X, Y=None, eval_gradient=False):
        if eval_gradient:
            raise ValueError("Gradient of non-smooth kernel does not exist.")
        X = np.atleast_2d(X)
        if Y is None:
            Y = X
        Y = np.atleast_2d(Y)
        return np.asarray([[ kron_delta(i, j) for i,j in zip(Xk, Yk) ]
                            for Xk,Yk in zip(X,Y)])
    
    def diag(self, X):
        return np.zeros((X.shape())) + 1


from scipy.optimize import minimize

def get_delta(a, b):
    """
    Find minimal delta for which |a-b| <= |a/e^delta - a|
    """
    diff = np.abs(a - b)
    obj = lambda x : np.abs( np.abs(a / np.e**x - a) - diff)
    return minimize(obj, 0.5, bounds=[(0, None)]).x[0]
        
print(get_delta(15 + 1j, 15 - 1j))


# parameters
n = 50
scale = 10
# generate random complex numbers with mean 0
Z = np.random.rand(n) + np.random.rand(n) * 1j
Z *= scale
Z -= sum(Z) / n
# any counterexamples?
for a in Z:
    for b in Z:
        if np.abs(np.log(b/a)) > get_delta(a, b) + 0.0001:
            print(a, b)
            assert(False)
print("good!")