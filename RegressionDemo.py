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

if __name__ == "__main__":
    # sample randomly from sine curve on [-4pi, 4pi]
    x = np.random.uniform(-4 * np.pi, 4 * np.pi, (50, 1))
    y = np.sin(x)

    # run regression
    gpr = GaussianProcessRegressor(
            kernel=ExpSineSquared(),
            n_restarts_optimizer=15)
    gpr.fit(x, y)
    
    # predict function
    x_pred = np.linspace(-8 * np.pi, 8 * np.pi, 101)[:, None]
    y_pred = gpr.predict(x_pred)
    
    # plot
    plt.scatter(x, y, color="black")
    plt.plot(x_pred, y_pred)
    plt.show()
        