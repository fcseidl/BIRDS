#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 09:41:40 2020

Generate sample trajectories from real-valued dynamical systems of arbitrary 
dimension.

@author: fcseidl
"""

import numpy as np
from scipy import linalg


class Diffeomorphism:
    r"""
    A smooth bijection |R^D -> |R^D. Generalizes 2D HMap class in KMEexample 
    module.
    """
    
    def __init__(self, D, rng):
        r"""
        Construct a Diffeomorphism object.
        
        Params
        ------
        D : dimension of space
        rng : np.random.RandomState
        """
        self.D = D
        # parameters 5 and -0.5 chosen somwhat arbitrarily
        self.polys = rng.rand(D - 1, 5) - 0.5
    
    def fwd(self, X):
        r"""
        Apply the forward diffeomorphism.
        
        Params
        ------
        X : array-like
            input values [v0, v1, ..., vk], with each vi in |R^D
        
        Returns
        -------
        Y : array with same shape as np.atleast_2d(X)
            contains fwd(v0), ..., fwd(vk).
        """
        X = np.atleast_2d(X)
        if X.shape[1] != self.D:
            raise ValueError("Expected X to contain vectors of length %s" 
                             % (self.D))
        Y = np.zeros(X.shape)
        for i in range(X.shape[0]):
            Y[i][:-1] = [ X[i][j] + np.polyval(self.polys[j - 1], X[i][0])
                        for j in range(1, self.D) ]
            Y[i][-1] = X[i][0]
        return Y
    
    def rvs(self, Y):
        r"""
        Apply the backward diffeomorphism.
        
        Params
        ------
        Y : array-like
            input values [v0, v1, ..., vk], with each vi in |R^D
        
        Returns
        -------
        X : array with same shape as np.atleast_2d(X)
            contains rvs(v0), ..., rvs(vk).
        """
        Y = np.atleast_2d(Y)
        if Y.shape[1] != self.D:
            raise ValueError("Expected X to contain vectors of length %s" 
                             % (self.D))
        X = np.zeros(Y.shape)
        for i in range(Y.shape[0]):
            X[i][0] = Y[i][-1]
            X[i][1:] = [ Y[i][j] - np.polyval(self.polys[j], Y[i][-1])
                        for j in range(0, self.D - 1) ]
        return X


class DynamicalSystem:
    r"""
    Implements a smooth dynamical system whose trajectories can be observed.
    Trajectories are images of an underlying linear system under a 
    polynomial.
    """
    
    def __init__(self, D, A=None, seed=42, sig=1e-4):
        r"""
        Construct a DynamicalSystem object.
        
        Params
        ------
        D : dimension of system
        A : array, optional
            matrix of dynamic map in linear coordinates, random by default.
        seed : random number generator seed, optional
            default value is 42
        sig : variance of additive Gaussian noise, optional
            default value is 1e-4
        """
        self.rng = np.random.RandomState(seed)
        self.D = D
        self.sig = sig
        if A is None:
            ## TODO: better random matrix, complex eigenvalues???
            A = self.rng.rand(D, D) * 2 - 1
        self.A = A
        # successive powers of dynamic map matrix A (initially A^0, A^1)
        self.Am = [np.identity(D), A]
        # maps between linear and observable coordinates
        h1 = Diffeomorphism(D, self.rng)
        h2 = Diffeomorphism(D, self.rng)
        self.fwd = lambda x : h1.fwd(h2.fwd(x))
        self.rvs = lambda x : h2.rvs(h1.rvs(x))
    
    def observe(self, N):
        r"""
        Observe a trajectory of length N with a random starting point.
        Returns the sequence of observed states in the trajectory 
        [x0, ..., xN-1]. Gaussian noise is added.
        """
        # compute needed powers of A
        while len(self.Am) < N:
            self.Am.append(np.dot(self.Am[-1], self.A))
        # random initial condition in linear coordinates
        y0 = self.rng.rand(self.D) - 0.5
        # trajectory in linear coordinates
        Y = np.dot(self.Am, y0)
        # trajectory in observation coordinates (with noise)
        return self.fwd(Y) + self.rng.randn(*Y.shape) * self.sig
    
    def eigen(self):
        r"""
        Return eigenvalues and associated normalized left-eigenvectors of the 
        dynamic map A in linear coordinates. 
        
        The eigenvalues of A are also Koopman eigenvalues: too see this, 
        note the dynamic map in observation coordinates is fwd(A*rvs), Thus 
        if U is the Koopman operator and vA = cv, then
        
        U v*rvs = v*rvs(fwd(A*rvs))
                = vA*rvs
                = cv*rvs,
                
        so v*rvs is an eigenfunction of U with eigenvalue c.
        """
        return linalg.eig(self.A, left=True, right=False)


if __name__ == "__main__":
    # plot a trajectory of a 3D system
    sys = DynamicalSystem(3, seed=5)
    traj = sys.observe(50)
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2])
    plt.show()
