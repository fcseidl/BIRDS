#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 17:47:28 2020

@author: fcseidl

Optimized Dynamic Mode Decomposition (DMD) via Gaussian process (GP) regression
based on Masuda et. al.: https://arxiv.org/pdf/1911.01143.pdf
"""

import numpy as np
import sklearn.gaussian_process as gp
from scipy import linalg


def masuda(Y, p=1, internals=False):
    r"""
    Params
    ------
    Y : array-like
        A sequence of N M-task observations of a dynamical system, 
        [ y_0, y_1, ..., y_N-1 ], y_i in |R^M
    p=1 : integer, optional
        Number of previous observations used to predict next observation.
        Input-output pairs used in training data are of the form (z_k, y_k),
        where z_k = [ y_k-p, y_k-p+1, ..., y_k-1 ].
    internals=False : bool, optional
        Whether or not to return additional internal features.
    
    Returns
    -------
    lams : array
        N - p approximate eigenvalues of Koopman operator U. These are called 
        Ritz values.
    modes : array
        modes[j] is the mode corresponding to the growth rate lams[j]. These 
        are called Ritz vectors.
    gpr : sklearn.gaussian_process.GaussianProcessRegressor
        Estimator which predicts y_k from input z_k. Returned only if internals 
        is True.
    Tgp_inv : array
        Inverse of Vandermonde matrix. Columns are eigenvectors of companion 
        matrix Cgp. Returned only if internals is True.
    """
    Y = np.atleast_2d(Y)
    # number of observations N
    N = Y.shape[0]
    if N <= p:
        raise ValueError(
                "p must be smaller than number of observations N")
    # training data for regression
    train_in = [ np.concatenate(Y[i:i + p]) for i in range(N - p) ]
    train_in = np.asarray(train_in)
    train_out = Y[p:]
    # covariance kernel
    # TODO: different scales and noise levels on different tasks?
    K = gp.kernels.ConstantKernel() * gp.kernels.RBF() \
            + gp.kernels.ConstantKernel() * gp.kernels.WhiteKernel()
    # run regression
    gpr = gp.GaussianProcessRegressor(kernel=K, n_restarts_optimizer=10)
    gpr.fit(train_in, train_out)
    # Ggp in |R^M x (N-p) holds predicted mean values of [ y_p, ..., y_N-1 ].
    Ggp = gpr.predict(train_in).T
    # build companion matrix Cgp in |R^(N-p) x (N-p)
    Kzz = K(train_in, train_in)
    zN = np.atleast_2d(np.concatenate(Y[-p:]))
    KzzN = K(train_in, zN)[:, 0]
    cgp = linalg.pinv(Kzz).dot(KzzN)
    Cgp = np.zeros((N - p, N - p))
    for i in range(N - p - 1): Cgp[i + 1][i] = 1
    Cgp[:, -1] = cgp
    # get Ritz values lams and Ritz vectors
    lams, Tgp_inv = linalg.eig(Cgp)   # Vandermonde matrix Tgp is Tgp_inv^-1
    Vgp = Ggp.dot(Tgp_inv)
    modes = Vgp.T
    if internals:
        return lams, modes, gpr, Tgp_inv
    return lams, modes


if __name__ == "__main__":
    from DataGeneration import DynamicalSystem
    M = 2   # dimension
    N = 60
    p = 13
    sys = DynamicalSystem(M)
    traj = sys.observe(N)
    print(masuda(traj, p))