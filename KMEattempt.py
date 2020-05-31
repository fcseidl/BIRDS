#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 18:45:43 2016

@author: fcseidl

Using Gaussian Process regression for Koopman Mode Estimation. Technique is 
from https://arxiv.org/pdf/1911.01143.pdf.
"""


if __name__ == "__main__":
    import numpy as np
    from scipy import linalg
    from DataGeneration import DynamicalSystem
    import DMDalgs
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.optimize import minimize, curve_fit
    import utilities as util
    from sklearn.metrics import r2_score
    
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
    N = 60
    p = 15  # p for masuda
    pp = 15 # p for sophist
    sys = DynamicalSystem(M, seed=1, sig=1e-6)
    #sys = DynamicalSystem(M, A=A, sig=1e-6)
    traj = sys.observe(N)
    
    print("...estimating modes and growth rates...")
    #lams, modes, gpr, T_inv, C = DMDalgs.masudaDMD(traj.T, p=p)
    #lams, modes, T_inv, C = DMDalgs.naiveProjectionDMD(traj)
    lams, modes, T_inv, C = DMDalgs.sophisticatedProjectionDMD(traj, p=pp)
    c = C[:, -1]
    k = len(lams)
    
    # estimate dynamic map T:|R^M -> |R^M. TODO: so far only works for p = 1
    T = lambda X : sys.fwd(np.dot(sys.A, sys.rvs(X)))
    # TODO: this currently uses hidden information. Could approximate T with 
    # supervised learning.
    
    task = 2 % M   # arbitrary
    
    # estimate matrix observable f(x_0) = [ x_0, ..., x_k-1 ]
    def f(X):
        """
        Params
        ------
        X : array
            Data matrix [ x_0, ... x_l ]
        
        Returns
        -------
        result : array
            such that result[i] = f_tilde(X[:, i]) in |R^k
        Xk : array
            Data matrix after k application of T,
            [ T^k(x_0), ..., T^k(x_l) ]
        """
        Xk = np.atleast_2d(X)
        result = []
        while len(result) < k:
            result.append(Xk[task])
            Xk = T(Xk)
        return np.asarray(result).T, Xk
    
    # an approximate eigenfunction in eigenspace of lams[i]
    def psi_i(i, X):
        """
        Apply psi_i to columns of X.
        """
        X = np.atleast_2d(X)
        F, _ = f(X)
        return np.dot(F, T_inv[:, i])
        
    # residual (error) of psi_i
    def res_i(i, X):
        """
        Apply res_i to columns of X.
        """
        X = np.atleast_2d(X)
        F, Xk = f(X)
        fk = Xk[task]
        # e_k-1(f^k - c dot f)
        return T_inv[-1][i] * (fk - np.dot(F, c))
    
    corcoeffs = []
    
    for i in range(k):
        vals = psi_i(i, traj)
        psi = vals[:-1]     # psi at each point except final
        Upsi = vals[1:]     # Koop op applied to psi
        
        mag = np.abs(psi)
        res = np.abs(Upsi - lams[i] * psi)
        #res_pred = np.abs(res_i(i, traj[:, :-1]))
        time = np.linspace(0, N - 2, N - 1)
        
        # exponential best fit
        def fun(x, a):
            return a * (np.abs(lams[i]) ** x)
        norm = linalg.norm(np.asarray(mag))
        a = [1]
        a, _ = curve_fit(fun, time, mag / norm, a)
        fit = [ fun(t, a[0]) * norm for t in time ]
        r2 = r2_score(mag, fit)
        corcoeffs.append(r2)
        
        fig, ax = plt.subplots()
        fig.suptitle("eigenfunction %s, eigenvalue %s" % (i, lams[i]))
        
        ax.set_ylim(bottom=0, top=1.2*max([max(mag), max(res), util.eps]))
        ax.set_xlabel("time step")
        ax.plot(time, mag, color='b', label="magnitude")
        ax.plot(time, res, color='r', label="residual magnitude")
        ax.plot(time, fit, color='g', linestyle="dashed",
                label=("exponential fit, r^2 = %s" % r2))
        #ax.plot(time, res_pred, color='g', 
         #       label="predicted residual magnitude")
        
        leg = plt.legend(
                loc="best", ncol=2, mode="expand", shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.5)
        
        fig.tight_layout()
        plt.show()
        
    
    fig, ax = plt.subplots()
    fig.suptitle("Eigenvalues")
    for i in range(k):
        ax.scatter(lams[i].real, lams[i].imag, 
                   color=(0, max(corcoeffs[i], 0), 0))
    ax.set_xlabel("real part")
    ax.set_ylabel("imaginary part")
    plt.show()
    
    
    '''    
    # performance readout
    print("i\t\t\t\tdelta\t\t\t\tlambda\t\t\t\t|lambda|")
    for i in range(k):  # index of eigenvalue
        deltas = []
        psi = psi_i(i, traj[0])
        for time in range(N - 1):
            Upsi = psi_i(i, traj[time + 1])   # Koop op on psi
            delta = np.abs(np.log(lams[i] * psi / Upsi))
            deltas.append(delta)
            psi = Upsi
        print(i, "\t", max(deltas), "\t", lams[i], "\t", np.abs(lams[i]))
    '''
    