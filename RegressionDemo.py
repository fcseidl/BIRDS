#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 12:05:14 2020

A demonstration of scikitlearn GP regressions.

@author: fcseidl
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared
from scipy.optimize import minimize


if __name__ == "__main__":
    N = 50
    # sample randomly from sine and cosine curves on [-4pi, 4pi]
    X = np.random.uniform(-4 * np.pi, 4 * np.pi, (N, 1))
    Y = [ np.sin(X)[:, 0], np.cos(X)[:, 0] ]
    Y = np.asarray(Y).T

    # run regression
    gpr = GaussianProcessRegressor(
            kernel=ExpSineSquared(),
            n_restarts_optimizer=15)
    gpr.fit(X, Y)
    
    # predict function
    X_pred = np.linspace(-8 * np.pi, 8 * np.pi, 101)[:, None]
    Y_pred = gpr.predict(X_pred)
    
    # plot
    plt.scatter(X, Y[:, 0], color="r")
    plt.scatter(X, Y[:, 1], color="b")
    plt.plot(X_pred, Y_pred[:, 0], color="r")
    plt.plot(X_pred, Y_pred[:, 1], color="b")
    plt.show()
