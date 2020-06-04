#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 17:47:28 2020

@author: fcseidl

Various functions for Dynamic Mode Decomposition (DMD). 
sophisticatedProjectionDMD appears most effective.
"""

import numpy as np
import sklearn.gaussian_process as gp
from scipy import linalg
from scipy.optimize import basinhopping
import utilities as util


def arnoldiLastStep(C, K):
    r"""
    Perform the final step of the Arnoldi-style algorithm proposed by 
    Rowley et. al., section 3: 
    http://cwrowley.princeton.edu/papers/koopman_jfm.pdf
    
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
    lams, T_inv = linalg.eig(C)   # Vandermonde matrix T is T_inv^-1
    V = K.dot(T_inv)
    return lams, V, T_inv


def masudaDMD(Y, p=1):
    r"""
    Perform DMD using technique from Masuda et. al.:
    https://arxiv.org/pdf/1911.01143.pdf
    
    Params
    ------
    Y : array-like
        Data matrix of N M-task observations of a dynamical system, 
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
        Estimator which predicts y_k from z_k.
    Tgp_inv : array
        Inverse of Vandermonde matrix. Columns are eigenvectors of companion 
        matrix Cgp.
    cgp : array
        Cgp[:, -1]. Approximate coefficients of projection of y_N-1 onto 
        y_0, ..., y_N-2.
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
    lams, Vgp, Tgp_inv, C = arnoldiLastStep(Cgp, Ggp)
    return lams, Vgp, gpr, Tgp_inv, C


def naiveProjectionDMD(Y):
    r"""
    Perform DMD obtaining final column of companion matrix as coefficients of 
    projection of Nth observation onto previous N - 1 observations.
    
    Params
    ------
    Y : array-like
        Data matrix of N M-task observations of a dynamical system, 
        [ y_0, y_1, ..., y_N-1 ], y_i = Y[:, i] in |R^M
    
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
    
    # minimize resto get coeffs of projection c = [c_0 ... c_N-2]
    callback = lambda x, f, accept : util.close(f, 0) and accept
    c = basinhopping(res, c, callback=callback).x

    # build companion matrix
    C = np.zeros((N - 1, N - 1))
    for i in range(N - 2): C[i + 1][i] = 1
    C[:, -1] = c
    
    # call Arnoldi algorithm to get eigenvalues and modes
    return arnoldiLastStep(C, Y[:, :-1])


def sophisticatedProjectionDMD(Y, p=15):
    r"""
    Perform DMD obtaining final column of companion matrix as coefficients of 
    projection of pth power of dynamic map onto previous p powers.
    
    Params
    ------
    Y : array-like
        Data matrix of N M-task observations of a dynamical system, 
        [ y_0, y_1, ..., y_N-1 ], y_i = Y[:, i] in |R^M
    p=15 : integer, optional
        Dimension of Krylov space to project onto.
    
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
    c = np.zeros(p)
    truth = Y[:, p:].T
    basis = [ Y[:, k:k+p] for k in range(N - p) ]
    
    # error norm to minimize
    # TODO: unweighted difference in norm biases against accuracy where state 
    # vector is small. Find empirically best norm.
    res = lambda x : linalg.norm(truth - np.dot(basis, x))
    ##res = lambda x : linalg.norm((truth - np.dot(basis, x)) / truth)
    
    # minimize res to obtain coeffs of projection c = [c_0 ... c_N-2]
    callback = lambda x, f, accept : util.close(f, 0) and accept
    c = basinhopping(res, c, callback=callback).x
    print("minimized norm from sophisticated projection =", res(c))
    
    # build companion matrix
    C = np.zeros((p, p))
    for i in range(p - 1): C[i + 1][i] = 1
    C[:, -1] = c
    
    # call Arnoldi algorithm to get eigenvalues and modes
    # choice of Y[:, :p] is arbitrary; any p consecutive observations would do
    return arnoldiLastStep(C, Y[:, :p])


# test projection techniques on a linear system
if __name__ == "__main__":
    M = 8   # dimension
    N = 10  # number of observations
    rng = np.random.RandomState(1)
    A = rng.rand(M, M) * 5 - 2.5
    x0 = rng.rand(M)    # start position
    
    # Data matrix
    Y = np.zeros((M, N))
    Y[:, 0] = x0
    for i in range(1, N):
        Y[:, i] = A.dot(Y[:, i - 1])
    
    # calculate eigenvalues 3 different ways: should all be close
    naive_lams, V, T_inv = naiveProjectionDMD(Y)
    soph_lams, V1, T_inv1 = sophisticatedProjectionDMD(Y, p=N-4)
    true_lams = linalg.eig(A)[0]
    naive_lams.sort()
    soph_lams.sort()
    true_lams.sort()
    print(naive_lams)
    print(soph_lams)
    print(true_lams)