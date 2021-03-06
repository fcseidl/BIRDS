#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 17:47:28 2020

@author: fcseidl

Various functions for Dynamic Mode Decomposition (DMD).
"""

import numpy as np
import sklearn.gaussian_process as gp
from scipy import linalg


def arnoldi(C, K, internals=False):
    r"""
    Params
    ------
    C : array
        Companion matrix.
    K : array
        Data matrix.
    
    Returns
    -------
    lams : array
        Ritz values.
    modes : array
        The column modes[:, i] is the Ritz vector corresponding to lams[i].
    T_inv : array
        Inverse of Vandermonde matrix. Columns are eigenvectors of companion 
        matrix C.   
    """
    lams, T_inv = linalg.eig(C)   # Vandermonde matrix Tgp is Tgp_inv^-1
    Vgp = K.dot(T_inv)
    modes = Vgp
    return lams, modes, T_inv


def masudaDMD(Y, p=1):
    r"""
    Perform DMD using technique from Masuda et. al. at 
    https://arxiv.org/pdf/1911.01143.pdf
    
    Params
    ------
    Y : array-like
        A sequence of N M-task observations of a dynamical system, 
        [ y_0, y_1, ..., y_N-1 ], y_i in |R^M
    p=1 : integer, optional
        Number of previous observations used to predict next observation.
        Input-output pairs used in training data are of the form (z_k, y_k),
        where z_k = [ y_k-p, y_k-p+1, ..., y_k-1 ].
    
    Returns
    -------
    lams : array
        N - p approximate eigenvalues of Koopman operator U. These are called 
        Ritz values.
    modes : array
        modes[:, j] is the mode corresponding to the growth rate lams[j]. These 
        are called Ritz vectors.
    gpr : sklearn.gaussian_process.GaussianProcessRegressor
        Estimator which predicts y_k from input z_k.
    Tgp_inv : array
        Inverse of Vandermonde matrix. Columns are eigenvectors of companion 
        matrix Cgp.
    cgp : array
        Cgp[:, -1]
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
    # call Arnoldi algorithm to get eigenvalues and modes
    lams, modes, Tgp_inv = arnoldi(Cgp, Ggp)
    return lams, modes, gpr, Tgp_inv, cgp


def naiveProjectionDMD(Y):
    r"""
    Perform DMD obtaining final column of companion matrix as coefficients of 
    projection of Nth observation onto previous N - 1 observations.
    
    Params
    ------
    Y : array-like
        A sequence of N M-task observations of a dynamical system, 
        [ y_0, y_1, ..., y_N-1 ], y_i in |R^M
    
    Returns
    -------
    lams : array
        N - 1 approximate eigenvalues of Koopman operator U. These are called 
        Ritz values.
    modes : array
        modes[:, j] is the mode corresponding to the growth rate lams[j]. These 
        are called Ritz vectors.
    T_inv : array
        Inverse of Vandermonde matrix. Columns are eigenvectors of companion 
        matrix C.
    """


if __name__ == "__main__":
    from DataGeneration import DynamicalSystem
    M = 2   # dimension
    N = 60
    p = 13
    sys = DynamicalSystem(M)
    traj = sys.observe(N)
    print(masuda(traj, p))