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
    import sklearn.gaussian_process as gp
    from DataGeneration import DynamicalSystem
    import DMDalgs
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.optimize import minimize
    import utilities as util
    
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
    N = 70
    p = 15  # p for masuda
    pp = 25 # p for sophist
    sys = DynamicalSystem(M, seed=0, sig=0)
    #sys = DynamicalSystem(M, A=A)
    traj = sys.observe(N)
    
    '''
    if M == 2:
        plt.plot(traj[:, 0], traj[:, 1])
        plt.show()
    if M == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2])
        plt.show()
    '''
    
    print("...estimating modes and growth rates...")
    #lams, modes, gpr, T_inv, C = DMDalgs.masudaDMD(traj.T, p=p)
    #lams, modes, T_inv, C = DMDalgs.naiveProjectionDMD(traj)
    lams, modes, T_inv, C = DMDalgs.sophisticatedProjectionDMD(traj, p=pp)
    c = C[:, -1]
    k = len(lams)
    
    # estimate dynamic map T:|R^M -> |R^M. TODO: so far only works for p = 1
    #T = lambda X : gpr.predict(X)
    T = lambda X : sys.fwd(np.dot(sys.A, sys.rvs(X)))
    
    # estimate matrix observable f(x_0) = [ x_0, ..., x_k-1 ]
    def f(X):
        X = np.atleast_2d(X)
        result = [X]
        while len(result) < k:
            result.append(T(result[-1]))
        return np.asarray(result).T
    
    # an approximate eigenfunction in eigenspace of lams[i]
    task = 1   # arbitrary
    def psi_i(i, X):
        X = np.atleast_2d(X)
        F = f(X)
        return np.dot(F[:, task], T_inv[:, i])
        
    # residual (error) of psi_i
    def res_i(i, X):
        X = np.atleast_2d(X)
        F = f(X)
        fk = T(F[:, :, -1].T)
        # e_k-1(f^k - c dot f)
        return T_inv[i, -1] * (fk[task] - np.dot(F[:, task], c))
    
    '''
    i = 0
    psi = psi_i(i, traj[0])
    for t in range(N - 1):
        Upsi = psi_i(i, traj[t + 1])
        actual = Upsi - lams[i] * psi
        expect = res_i(i, traj[t])
        # TODO: why is actual so much bigger than expect?!?!?
        2 + 2   # dummy line
    '''
    
    for i in range(k):
        vals = psi_i(i, traj)
        psi = vals[:-1]     # psi at each point except final
        Upsi = vals[1:]     # Koop op applied to psi
        
        # residual graph
        mag = np.abs(psi)
        res = np.abs(Upsi - lams[i] * psi)
        res_pred = np.abs(res_i(i, traj[:, :-1]))
        time = np.linspace(0, N - 2, N - 1)
        
        fig, ax = plt.subplots()
        fig.suptitle("eigenfunction %s, eigenvalue %s" % (i, lams[i]))
        
        ax.set_ylim(bottom=0, top=1.2*max(max(mag), max(res)))
        ax.set_xlabel("time step")
        ax.plot(time, mag, color='b', label="magnitude")
        ax.plot(time, res, color='r', label="residual magnitude")
        ax.plot(time, res_pred, color='g', 
                label="predicted residual magnitude")
        
        leg = plt.legend(
                loc="best", ncol=2, mode="expand", shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.5)
        
        fig.tight_layout()
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
    