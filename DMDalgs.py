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
from scipy.optimize import basinhopping
import utilities as util


def arnoldi(C, K):
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
    V : array
        The column V[:, i] is the Ritz vector corresponding to lams[i].
    T_inv : array
        Inverse of Vandermonde matrix. Columns are eigenvectors of companion 
        matrix C.  
    """
    lams, T_inv = linalg.eig(C)   # Vandermonde matrix Tgp is Tgp_inv^-1
    V = K.dot(T_inv)
    return lams, V, T_inv


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
    Vgp : array
        Vgp[:, j] is the mode corresponding to the growth rate lams[j]. These 
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
    lams, Vgp, Tgp_inv = arnoldi(Cgp, Ggp)
    return lams, Vgp, gpr, Tgp_inv, cgp


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
    Y = np.atleast_2d(Y)
    N = Y.shape[1]
    c = np.zeros(N - 1)
    # length of residual (c_0 * y_0 + ... + c_N-2 * y_N-2) - y_N-1
    res = lambda x : linalg.norm(np.dot(Y[:, :-1], x) - Y[:, -1])
    # coeffs of projection c = [c_0 ... c_N-2]
    callback = lambda x, f, accept : util.close(f, 0) and accept
    c = basinhopping(res, c, callback=callback).x
    # Companion matrix
    C = np.zeros((N - 1, N - 1))
    for i in range(N - 2): C[i + 1][i] = 1
    C[:, -1] = c
    # call Arnoldi algorithm to get eigenvalues and modes
    return arnoldi(C, Y[:, :-1])


if __name__ == "__main__":
    M = 20
    N = 15
    rng = np.random.RandomState(0)
    A = rng.rand(M, M) * 5 - 2.5
    x0 = rng.rand(M)
    Y = np.zeros((M, N))
    Y[:, 0] = x0
    for i in range(1, N):
        Y[:, i] = A.dot(Y[:, i - 1])
    lams, V, T_inv = naiveProjectionDMD(Y)
    print(lams)
    print(linalg.eig(A)[0])