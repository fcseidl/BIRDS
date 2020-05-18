# -*- coding: utf-8 -*-
"""
Created on Mon May  2 18:45:43 2016

@author: shrevz

Using (naive) Gaussian Process estimator to compute linearizing observable for
a system whose trajectories can be observed
"""
from numpy import (
  dot,newaxis,sign,asfarray,exp,log,sqrt,arange,
  zeros,concatenate,asarray,polyval,identity,sum,cumsum)
from scipy.linalg import pinv, inv, svd
from pylab import (randn,rand,randint, norm)

#### a diffeomorphism?
class HMap(object):
  def __init__(self):
    self.p = rand(5)-0.5

  def fwd(self,xy):
    x,y = asarray(xy)
    x.shape = (1,)+x.shape
    y.shape = (1,)+y.shape
    return concatenate([y+polyval(self.p,x),x],0)

  def rvs(self,xy):
    x,y = asarray(xy)
    x.shape = (1,)+x.shape
    y.shape = (1,)+y.shape
    return concatenate([y,x-polyval(self.p,y)],0)

if __name__ == "__main__":
    ###
    ### Note flip K entries based on signum cf
    ###
    try:
        #### reuse data if cached
      assert notaname
      assert type(X)==list and X[0].shape[0] == D
      print("Using",len(X),"cached trials of lengths",[Xk.shape[1] for Xk in X])
    except NameError:
        #### data not cached, generate new
        
        #### Parameters specified in advance
      # Dimension
      D = 2
      # Maximal trial length
      Nm = 4
      # Number of trials
      N = 10
      
      #### generate a linear transformation A; 
      #### A is determined by eigenvalues and eigenvectors
      ##lams = asfarray([-0.05,-0.15])
      # Eigenvalues
      lams = asarray([-0.05+.1j, -0.05-.1j])
      mus = exp(lams)
      #### mus like the letter mu; expectation?
      # Eigenvectors #### (unit length?)
      V = randn(D,D)
      V /= sqrt(sum(V*V,0))[newaxis,:]
      # Matrix from eigenvalues and eigenvectors
      if all(mus == mus.real):
        A = dot(inv(V),mus[:,newaxis]*V)
      else:
        mr = mus[0].real
        mi = mus[0].imag
        A0 = asarray([[mr, mi],[-mi, mr]])
        A = dot(inv(V),dot(A0,V))
        
      # Length of trials
      Nk = [randint(int(log(Nm)/log(2)),Nm) for k in range(N)]
      # Trials in linear coordinates
      Y = []
      print("Creating",N,"trials:", end=' ')
      # Create list of multiplier matrices
      Am = [identity(D)]
      for kk in range(Nm-1):
        Am.append(dot(A,Am[-1]))
      Am = concatenate( Am, 0 )
      for nn in Nk:
        # Randmize initial condition
        x0 = rand(D)-0.5
        # Compute all multipliers
        y0 = dot(Am,x0).reshape(Nm,D)
        Y.append(y0[:nn,:].T)
        print(nn, end=' ')
      print(" DONE")
      # Create coordinate change
      fwd = lambda x : x
      rvs = lambda x : x
      if 1:
        h1 = HMap()
        h2 = HMap()
        fwd = lambda x : h1.fwd(h2.fwd(x))
        rvs = lambda x : h2.rvs(h1.rvs(x))
      # Observed trajectories in observation coordinates
      X = [ fwd(Yk) + randn(*Yk.shape)*1e-4 for Yk in Y ]
      #### randn(*Yk.shape)*1e-4 is additive noise
    
      ###
      ### Algorithm parameters that depend ONLY on data point locations
      ###   but not on mu or choice of kernel
      ###
      if 1:
        # Compute matrix size
        NN = sum(Nk)
        # All data matrix
        Xall = concatenate(X,axis=1).T
        # Distance (squared) matrix
        Xdst = zeros((NN,NN),float)
        for k,xk in enumerate(Xall):
          dk = sum((xk[newaxis,...] - Xall)**2,1)
          Xdst[k,:] = dk
          Xdst[:,k] = dk
    
      # compute pairwise trajectory distance
      # --> this will be used as distance when passed to GP kernel
      if 1: # nearest approach distances
        # Offset trial in Xall
        CNk = cumsum(Nk)
        # Compute nearest approach distance between trajectories
        # Index by Xall entry
        Dx = zeros((NN,NN),float)
        # Index by trial number
        Dc = zeros((N,N),float)
        s0 = 0
        for n0,e0 in enumerate(CNk):
          for n1,e1 in enumerate(CNk):
            if n1<n0:
              continue
            if n1==n0:
              s1=s0
              continue
            d = Xdst[s0:e0,s1:e1].min()
            Dx[s0:e0,s1:e1] = d
            Dx[s1:e1,s0:e0] = d
            Dc[n0,n1] = d
            Dc[n1,n0] = d
            s1 = e1
          s0 = e0
    
    # There is data ready
    assert type(X)==list and X[0].shape[0] == D
    
    #### now we have data
    
    
    if 1:
      # Pick index of mu to use
      muk = 0
      mu = mus[muk]
      # Ground truth solution
      print("... building ground truth ...")
      Xi = asarray([Xk[:,0] for Xk in X])
      Cref = dot(rvs(Xi.T).T,V[muk])
      Cref /= norm(Cref)
      psiref = dot(rvs(Xall.T).T,V[muk])
    
    if 1: # F matrix alg
      # F matrix
      print("... building trial Vandermond (F) matrix ...")
      F = zeros((NN,N),complex)
      r = 0
      for k,Xk in enumerate(X):
        n = Xk.shape[1]
        F[r:r+n,k] = mu**arange(n)  #### how can we know mu already?!?!?
        r += n
      ## Kernel matrix
      print("... building (unsigned) kernel ...")
      ##   Meta-parameter l = 1
      #K0 = exp(-Xdst/(2*0.1)) # fails
      #K0 = exp(-sqrt(Xdst)/0.5) # Kinda works
      #K0 = exp(-Dx/2) # Kinda works
      K0 = 1/(0.2+Xdst) # Smash hit
      #K0 = 1/(1+Dx)
      # Matrix to solve for C
      M0 = dot(dot( F.T, pinv(K0,1e-13) ), F)
      ###M0 = pinv(dot(dot( F.T, K0),F ),1e-13)
      # Solving for C
      Um0,Sm0,Vm0 = svd(M0)
      # Solution
      C0 = Vm0[-1,:]
      psi0 = dot(C0,F.T)
    
    from pylab import figure, draw, gcf, title
    if 0: # plot result
        from mpl_toolkits.mplot3d import Axes3D
        for k in range(5):
          # Solution
          figure(101+k)
          Ck = Vm[-k-1,:]
          psik = dot(Ck,F.T)
          psik *= sign(dot(psik,psiref))
          Ck0 = Vm0[-k-1,:]
          psik0 = dot(Ck0,F.T)
          psik0 *= sign(dot(psik0,psiref))
          ax = gcf().add_subplot(111, projection='3d')
          ax.scatter(Xall[:,0],Xall[:,1],psik,color='b')
          ax.scatter(Xall[:,0],Xall[:,1],psi0,color='r')
          ax.scatter(Xall[:,0],Xall[:,1],psik0,color='g')
          title('SVD #%d (%.3e)' % (k+1, dot(dot(Ck,M),Ck)))
          draw()
    
    if 1: # Compare corrections
      p= Cref>0
      cp = Cref[p].copy(); cp = cp / norm(cp)
      v0p = Vm0[-1,p].copy(); v0p = v0p / norm(v0p) * sign(dot(v0p,cp))
      figure(44).clf()
      ax = gcf().add_subplot(211)
      ax.plot( cp, '.' )
      ax.plot( v0p, '+' )
      cn = Cref[p].copy(); cn = cn/norm(cn)
      v0n = Vm0[-1,p].copy(); v0n = v0n / norm(v0n) * sign(dot(v0n,cn))
      ax = gcf().add_subplot(212)
      ax.plot( cn, '.' )
      ax.plot( v0n, '+' )
    
    if 1: # plot result
        from mpl_toolkits.mplot3d import Axes3D
        Axes3D
        for k in range(2):
          # Solution
          figure(101+k)
          ax = gcf().add_subplot(111, projection='3d')
          Ck0 = Vm0[-k-1,:].copy(); Ck0 *= sign(dot(Ck0,Cref))
          for ck0,c0,Xk in zip(Ck0,Cref,X):
            mk = mu**arange(Xk.shape[1])
            ax.plot3D(Xk[0],Xk[1],c0*mk,'.-r')
            ax.plot3D(Xk[0],Xk[1],ck0*mk,'x-g')
        ax.axis('equal')
    
    if 1:
      fig = figure(99); fig.clf();
      mm = [Cref.min(), Cref.max()]
      ax = fig.add_subplot(111)
      ax.plot( Cref, Vm0[-1,:], '+m', mew=2,ms=10 )
      ax.plot( Cref, Vm0[-2,:], 'xr', mew=2 ,ms=10)
      ax.plot( mm, mm, 'k')
      ax.plot( mm, mm[::-1], 'k')
      ax.axis('equal'); ax.grid(1)
    
    show()