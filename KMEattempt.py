#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 18:45:43 2016

@author: fcseidl

Using Gaussian Process regression for Koopman Mode Estimation. Technique is 
from https://arxiv.org/pdf/1911.01143.pdf.
"""

from scipy.optimize import minimize

def get_delta(a, b):
    """
    Find minimal delta for which |a - b| <= |a/e^delta - a|
    """
    # normalize a and b
    c = (a + b) / 2
    a /= c
    b /= c
    # now find delta
    diff = np.abs(a - b)
    obj = lambda x : np.abs( np.abs(a / np.e**x - a) - diff)
    return minimize(obj, 0.5, bounds=[(0, None)]).x[0]


if __name__ == "__main__":
    import numpy as np
    from scipy import linalg
    import sklearn.gaussian_process as gp
    from DataGeneration import DynamicalSystem
    from MasudaDMD import masuda
    import matplotlib.pyplot as plt
    
    M = 3 # dimension
    
    '''
    mus = np.exp(np.asarray([-0.05+.1j, -0.05-.1j]))
    # Eigenvectors
    V = np.random.randn(M,M)
    V /= np.sqrt(sum(V*V,0))[np.newaxis,:]
    # Matrix from eigenvalues and eigenvectors
    if all(mus == mus.real):
        A = np.dot(linalg.inv(V),mus[:,np.newaxis]*V)
    else:
        mr = mus[0].real
        mi = mus[0].imag
        A0 = np.asarray([[mr, mi],[-mi, mr]])
        A = np.dot(linalg.inv(V),np.dot(A0,V))
    '''
    
    # get sample trajectory from random dynamical system
    N = 40
    sys = DynamicalSystem(M, seed=3)
    #sys = DynamicalSystem(M, A=A)
    traj = sys.observe(N)
    if M == 2:
        plt.plot(traj[:, 0], traj[:, 1])
        plt.show()
    if M == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2])
        plt.show()
    
    # estimate modes and growth rates
    lams, modes, gpr, Tgp_inv = masuda(traj, internals=True)
    
    # estimate dynamic map F:|R^M -> |R^M. TODO: so far only works for p = 1
    F = lambda x : gpr.predict(np.atleast_2d(x))
    # estimate matrix observable f(x_0) = [ x_0, ..., x_N-1 ]^T
    def f(x):
        result = [[x]]
        while len(result) < N - 1:
            result.append(F(result[-1]))
        return np.asarray(result)[:, 0]
    # an approximate eigenfunction
    psi = lambda i, x : Tgp_inv[:, i].dot(f(x)[:, 0])
    
    # measure how well estimate performed
    print("\nEstimated eigenfunction psi appears to be in the", 
          "(lambda, |Upsi/e^delta - Upsi|)-pseudospectrum.")
    print("delta\t\t\t\tlambda")
    for i in range(N - 1):
        residuals = []
        deltas = []
        ratios = []
        psi_i = psi(i, traj[0])
        for j in range(N - 1):
            Upsi = psi(i, traj[i + 1])     # Koopman operator applied to psi
            residuals.append( np.abs( Upsi - lams[i] * psi_i ) )
            deltas.append(get_delta(Upsi, lams[i] * psi_i))
            ratios.append(lams[i] * psi_i / Upsi)
            if np.abs(np.log(lams[i] * psi_i / Upsi)) > deltas[-1] + 0.001:
                assert(False)
            psi_i = Upsi
        #print("residuals:\n", residuals)
        #print("ratios:\n", ratios)
        #print("deltas:\n", deltas)
        print(max(deltas), "\t\t", lams[i])
    
