#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 17:20:27 2020

@author: fcseidl
"""

# sample random points for training data
import numpy as np
rng = np.random.RandomState(42)
X_train = rng.rand(500, 2) * 20 - 10
phi_train = X_train[:, 0] ** 2 + X_train[:, 1] ** 2

# run regression
from sklearn.gaussian_process import GaussianProcessRegressor
gpr = GaussianProcessRegressor()
gpr.fit(X_train, phi_train)

# plot predicted data
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter3D(X_train[:, 0], X_train[:, 1], phi_train);
X_plot = np.array(np.meshgrid(np.linspace(-15, 15, 50),
                              np.linspace(-15, 15, 50))).T.reshape(-1, 2)
phi_pred = gpr.predict(X_plot)
ax.plot_trisurf(X_plot[:, 0], X_plot[:, 1], phi_pred, color='red')

# show plot
ax.set_xlabel('y')
ax.set_ylabel('v')
ax.set_zlabel('phi')
fig.show()

print(gpr.log_marginal_likelihood())

'''
# now plot truth
ax.clear()
ax.set_xlabel('y')
ax.set_ylabel('v')
ax.set_zlabel('phi')
phi_true = X_plot[:, 0] ** 2 + X_plot[:, 1] ** 2
ax.plot_trisurf(X_plot[:, 0], X_plot[:, 1], phi_true, color='blue')
'''
    