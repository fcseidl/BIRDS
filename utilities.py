#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 10:03:22 2020

@author: fcseidl

Various helper classes and functions for Koopman mode estimation.
"""

import numpy as np
from scipy.stats import linregress

eps = 1e-6


def exponential_regression(x, y):
    r"""
    Params
    ------
    x, y : array-like
        length-n arrays of x and y values.
    
    Returns
    -------
    a, b : numeric
        such that ae^bx is the (least squares) exponential curve of best fit 
        to x and y.
    r2 : numeric
        correlation coefficient (of linear regression to log data)
    """
    b, inter, r2, _, __ = linregress(x, np.log(y))
    a = np.e ** inter
    return a, b, r2
    

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