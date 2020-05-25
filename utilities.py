#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 10:03:22 2020

@author: fcseidl

Various helper classes and functions for Koopman mode estimation.
"""

import numpy as np
import sklearn.gaussian_process as gp
from scipy.optimize import minimize

eps = 1e-6


def proj(u, v):
    r"""
    Return projection of u onto v.
    """
    return u.dot(v) / v.dot(v) * v


def close(a, b, epsilon=eps):
    r"""
    Return whether two numbers are close.
    """
    return np.abs(a - b) < epsilon


if __name__ == "__main__":
    u = np.asarray([1, 2, 3])
    v = np.asarray([5, 6, 2])
    print(proj(u, v))
    print(proj(v, u))
    print(proj(u, u))
    print(proj(v, v))


'''
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
'''