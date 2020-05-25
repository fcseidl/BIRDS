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
    N = 50
    p = 1
    sys = DynamicalSystem(M, seed=6, sig=0)
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
    
    # estimate modes and growth rates
    #lams, modes, gpr, T_inv, c = DMDalgs.masudaDMD(traj, p=p)
    lams, modes, T_inv = DMDalgs.naiveProjectionDMD(traj)
    
    # estimate dynamic map T:|R^M -> |R^M. TODO: so far only works for p = 1
    #T = lambda x : gpr.predict(x)
    T = lambda x : sys.fwd(np.dot(sys.A, sys.rvs(x)[0]))
    # estimate matrix observable f(x_0) = [ x_0, ..., x_N-2 ]^T
    def f(x):
        result = [np.asarray([x])]
        while len(result) < N - p:
            result.append(T(result[-1]))
        return np.asarray(result)[:, 0]
    # an approximate eigenfunction in eigenspace of lams[i]
    psi_i = lambda i, x : T_inv[:, i].dot(f(x)[:, 0])
    
    
    # graph of psi_i
    i = 14
    time = []
    res = []
    mag = []
    psi = psi_i(i, traj[0])
    for t in range(N - 1):
        Upsi = psi_i(i, traj[t + 1])
        time.append(t)
        res.append(np.abs(Upsi - lams[i] * psi))
        mag.append(np.abs(psi))
        psi = Upsi
    
    fig, ax1 = plt.subplots()
    
    color = 'b'
    ax1.set_ylim(bottom=0, top=1.2*max(max(mag), max(res)))
    ax1.set_xlabel("time step")
    ax1.set_ylabel("magnitude of eigenfunction",
                   color=color)
    ax1.plot(time, mag, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    color = 'r'
    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_ylabel("magnitude of residual", color=color)
    ax2.plot(time, res, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.show()
        
    
    # performance readout
    print("i\t\t\t\tdelta\t\t\t\tlambda\t\t\t\t|lambda|")
    for i in range(N - 1):  # index of eigenvalue
        deltas = []
        psi = psi_i(i, traj[0])
        for time in range(N - 1):
            Upsi = psi_i(i, traj[time + 1])   # Koop op on psi
            delta = np.abs(np.log(lams[i] * psi / Upsi))
            deltas.append(delta)
            psi = Upsi
        print(i, "\t", max(deltas), "\t", lams[i], "\t", np.abs(lams[i]))
    