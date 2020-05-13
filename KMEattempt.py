#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 18:45:43 2016

@author: shrevz

Using (naive) Gaussian Process estimator to compute linearizing observable for
a system whose trajectories can be observed.

@modified: fcseidl, May 2020

Now featuring hyperparameter selection.
"""

import numpy as np
import sklearn.gaussian_process as gp
from DataGeneration import DynamicalSystem
    

if __name__ == "__main__":
    # dimension
    M = 2
    # number of observations
    N = 15
    # predict y_n from [y_(n-p), ..., y_(n-1)]
    p = 3
    
    sys = DynamicalSystem(M)
    # compute trajectory
    traj = sys.observe(N).T
    # training input
    Z = [ np.concatenate(traj[i:i + p]) for i in range(N - p) ]
    Z = np.asarray(Z)
    # training output
    Y = traj[p:]
    
    # compute scaling parameters and rescale training data
    s = np.zeros((M * p,))
    for i in range(M * p):
        s[i] = max(Z[:, i])
        Z[:, i] *= s[i]
    ss = np.zeros((M,))
    for i in range(M):
        ss[i] = max(Y[:, i])
        Y[:, i] *= ss[i]
    
    # compute intertask covariance matrix Kg
    Kg = np.zeros((M, M))
    for i in range(M):
        Kg[i][i] = ss[i] ** -1
    
    # create Gaussian kernel
    ker = (gp.kernels.ConstantKernel() * gp.kernels.RBF() 
            + gp.kernels.WhiteKernel())
    
    # perform fit
    gpr = gp.GaussianProcessRegressor(kernel=ker)
    gpr.fit(Z, Y)  
    
    
