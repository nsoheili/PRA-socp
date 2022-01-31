#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRA second-order code -- algorithms only
Modified on Tuesday June 15, 2021

@authors: Javier Peña and Negar Soheili
"""
import numpy as np
import scipy
import scipy.linalg
from scipy.sparse import csr_matrix
import cvxpy as cp

class Cone():
    """ 
    Product of second order cones
    """
    def __init__(self,dim):
        self.dim = dim
        self.r = len(dim)
        sumdim = np.insert(self.dim.cumsum(),0,0)
        barcomp = np.ndarray((0,sumdim[-1]))
        self.sdim = np.delete(sumdim,len(sumdim)-1)
        for i in range(self.r):
            indx = np.arange(self.dim[i])+self.sdim[i]   
            rowi = np.zeros(sum(self.dim))
            rowi[indx[1:]] = 1
            barcomp = np.vstack((barcomp,rowi))
        self.barcomp = csr_matrix(barcomp)  # matrix to extract the "bar" components of a vector in the cone

    def eigenvalues(self,x):
        if len(x)!=np.sum(self.dim):
            print('mismatch dimensions')
            return
        else:
            norms = (self.barcomp@(x**2))**0.5  # norms of the bar components
            eigval = np.kron(x[self.sdim],[1,1])+np.kron(norms,[-1,1])
            normalizer = self.barcomp.T@norms
            normalizer[normalizer > 0] = 1/normalizer[normalizer>0]
            eigvec = x*normalizer
            eigvec[self.sdim] = 1
        return eigval,eigvec

    def Umu(self,g,mu,x0):
        # Finds armgmin g'x + 0.5*mu*\|x-x0\|^2 over the secondplex
        gg = g/mu - x0
        eigval,eigvec = self.eigenvalues(gg)
        ll = softmax(-eigval)
        u = np.diag(self.barcomp.T@(np.kron(np.eye(self.r),[-1,1])@ll))@eigvec
        u[self.sdim] = np.kron(np.eye(self.r),[1,1])@ll
        return u/2

def PRA(A, AA, K, z0, aggressive = True, RescalingLimit = 50):
    """ Projection and Rescaling Algorithm
     *Inputs:*
     A and AA       : Matrices such that L = null(A) and Lperp = null(AA) 
     K              : Direct product of second-order cones
     z0             : Initial point in K
     aggressive     : Boolean variable to enable agressive rescaling heuristic
     RescalingLimit : Upper bound on the number of rescaling rounds 
    
     *Outputs:*
     feas           : Feasibility status of the problem identified by the algorithm.
                      feas = 1 when found an interior point xL in L \cap K
                      feas = 2 when found an interior point xLperp in L \cap K
                      feas = -1 when reached RescalingLimit without solving either problem
     xL             : Solution found in L\cap K
     xLperp         : Solution found in Lperp \cap K
     k              : Number of rescaling rounds
     Total          : Total number of iterations
    """
    Q,R = np.linalg.qr(AA.T) 
    P = Q@Q.T

    QQ, R = np.linalg.qr(A.T)
    PP = QQ@QQ.T 

    # Initialization*
    Total = 0; k = 0; feas = 0 
    zz0 = z0 
    D = np.eye(len(z0)); DD = D 

    while (feas == 0) and (k <= RescalingLimit):   
        # ** Basic procedure phase (smooth perceptron algorithm)
        # Primal
        y, z, totalitrsmooth, feasprimal = basic(K, P, z0, 0.2) 
        Total += totalitrsmooth         
        # Dual 
        yLperp, zLperp, totalitrsmooth,feasdual = basic(K, PP, zz0, 0.2) 
        Total += totalitrsmooth 

        # ** Check if basic procedure succeeded
        if feasprimal:
            feas = 1
            xL = np.linalg.solve(D,y) 
            xL = xL/np.linalg.norm(xL)  # normalize 
            xLperp = xL*0
            return feas,xL,xLperp,k,Total
        if feasdual:
            feas = 2
            xLperp = np.linalg.solve(DD,yLperp)
            xLperp = xLperp/np.linalg.norm(xLperp)  # normalize 
            xL = xLperp*0
            return feas,xL,xLperp,k,Total
    
        # ** Rescaling phase -- primal 
        B = rescale(K,P,z,aggressive)            
        D = B@D        
        # Update the projection matrix after rescaling
        Q,R = np.linalg.qr(D@AA.T)
        P = Q@Q.T 
        # ** Rescaling phase -- dual
        B = rescale(K,PP,zLperp,aggressive)
        DD = B@DD 
        # Update the projection matrix after rescaling       
        QQ,R = np.linalg.qr(DD@A.T) 
        PP = QQ@QQ.T
        k = k+1 
    if (k > RescalingLimit):
        print('Could not finish within '+str(RescalingLimit)+' iterations')
        feas = -1 ; xL = 0*z0; xLperp = 0*z0
        return feas,xL,xLperp,k,Total

def rescale(K,P,z,aggressive):
    """ Compute rescaling matrix
    """
    Pzplus,ev = K.eigenvalues(P@z)
    Pzplus = Pzplus*(Pzplus>0)
    onePz = np.sum(Pzplus)
    B=np.ndarray((0,0))
    for i in range(K.r):
        # Update the rescaling matrix in i-th block
        indx = np.arange(K.dim[i])+K.sdim[i]
        eps = onePz/z[K.sdim[i]]
        if eps < 1:
            if aggressive:  # rescale aggressively
                eps = eps/K.r 
            zblock = z[indx]/z[K.sdim[i]]         
            Bblk = Bblock(zblock,eps)
            B = scipy.linalg.block_diag(B,Bblk)
        else:
            B = scipy.linalg.block_diag(B,np.eye(K.dim[i]))                
    return B

def Bblock(z,eps):
    # Construct the ith block of the rescaling matrix
    a = ((2/eps-1)**0.5-1)/2
    v = a*z; v[0] = v[0]+1
    v = v/((1+2*a*(a+1)*eps)**(0.5))        
    R = -np.eye(len(v))
    R[0,0]=1
    B = 2*np.tensordot(v, v, axes=0)-(v[0]**2-np.dot(v[1:],v[1:]))*R
    return B

def basic(K,P,u0,epsilon=0.5):
    """
    Smooth perceptron for a second-order conic system L \cap K
    *Inputs:*
     P       : The projection matrix onto L 
     u0      : Initial solution in Delta(K), 
     epsilon : An upper bound on the rescaling condition in ||(Pz)+||_1/max(z)<= epsilon ;
    
    *Output:*
     y  : Pu,
     z  : A solution satisfying either Pz in int(K) or the rescaling condition
          sum(max(Pz,0)) <= epsilon*lmax(z),
     k  : Number of iterations taken by the smooth perceptron algorithm 
     feas: binary variable to indicate if y is in the cone
    """
    # Initialization*
    k = 0 ; mu = 2; u = u0 ;
    y = P@u; z = K.Umu(y,mu,u0); w = P@z 
    lw,ew = K.eigenvalues(w) ;  lz,ew = K.eigenvalues(z) ; ly,ey = K.eigenvalues(y)
    # Smooth perceptron updates
    while (np.sum(lw*(lw>0))/np.max(lz) > epsilon) and (np.sum(ly < 0) > 0):     
        theta = 2/(k+3);
        u = (1-theta)*(u+theta*z) + (theta**2)*K.Umu(y,mu,u0) 
        mu = (1-theta)*mu
        y = P@u  
        z = (1-theta)*z + theta*K.Umu(y,mu,u0)
        w = P@z 
        lw,ew = K.eigenvalues(w) ; lz,ew = K.eigenvalues(z) ; ly,ey = K.eigenvalues(y)
        k = k+1  
    if k > 0:
        k = k-1
    feas = (np.sum(ly<0) == 0)
    return y,z,k,feas

def softmax(g):
    """
    Finds argmin 0.5*\|x\|^2-g'*x over the standard simplex
    Think of x as a 'softmax' of g.  The vector x is a discrete distribution:
        x = (g-lmin)^+ where lmin is so that sum(x) == 1.
    The vector x is concentrated on a single component if that component
    of g is sufficiently larger than all of the others.
    """
    lmin = np.max(g) - 1
    lmax = lmin + (np.sum((g>lmin)*(g-lmin))-1)/np.sum(g>lmin)
    while lmax > lmin+1e-10 :
        lmin = lmax - (1-np.sum((g>lmax)*(g-lmax)))/np.sum(g>lmax)
        lmax = lmin + (np.sum((g>lmin)*(g-lmin))-1)/np.sum(g>lmin)
    x=np.where(g>lmin,g-lmin,0)
    return x

def NaiveInstance(m,n):
    """ 
    Generate random A and AA naively so that null(A) and null(AA) 
    are orthogonal complements of each other.
    """
    M = np.random.randn(n,n)
    q,r = np.linalg.qr(M)
    A = q[0:m,:]
    AA = q[m:n,:]
    return A, AA   

def ControlledInstance(m, x, K):
    """
    Generate random instances with controlled condition measures
    Given  x \in K, "matrix_controlledcondition" generates instance A
    such that x is the most interior solutions to null(A)\cap K.
    Generate also AA so that null(AA) is the orthogonal complement of null(A)
    *Inputs:*
     m : Number of rows ;
     n : Number of columns ;
     x : The most interior solution. Assume \|x\|_\inf = 1 ;

     *Outputs:*
     A: random balanced matrix A such that Ax = 0 ;
     AA: balanced matrix such that null(AA) is the orthogonal complement of null(A)
    """
    n = len(x)
    r = K.r
    A = np.random.randn(m-1,n) 
    A = A - np.tensordot(A@x,x,axes=0)/(x.T@x) 
    lx,ev = K.eigenvalues(x) 
    if (np.min(lx)<=0):
        print('provided x is not in the cone')
        return
    # compute norms of blocks
    nx = np.kron(np.eye(r),np.ones(2))@lx**2
    lu = nx*0
    indx = (nx==np.max(nx)).nonzero()[0] 
    lu[indx] = np.random.rand(len(indx)) 
    lu = K.r*lu/sum(abs(lu))
    u = np.zeros(n)
    xinv = -x 
    xinv[K.sdim]=x[K.sdim]
    for i in range(K.r):
        indx = np.arange(K.dim[i])+K.sdim[i]
        u[indx] = x[indx]*lu[i]
        xinv[indx]=xinv[indx]/(xinv[indx[0]]**2-np.dot(xinv[indx[1:]],xinv[indx[1:]]))    
    A = np.vstack((A,u.T-xinv.T))
    Q,R = np.linalg.qr(A.T,mode='complete') ;
    A = Q[:,0:m].T 
    B = Q[:,m:n].T 
    return A,B
        
def PRAcpversion(A,AA,K,solver = None):
    """
    This functions does the same as PRA via cvxpy for comparison purposes.
    May use any solver available in the cvxpy installation.
    """
    n = sum(K.dim)
    x = cp.Variable(n)

    # Set up the SOCP formulations in cvxpy and try to solve them

    # First, for L \cap K
    soc_constraints = [cp.SOC(x[K.sdim[i]] - 1, x[K.sdim[i]+1:K.sdim[i]+K.dim[i]]) for i in range(K.r)]
    prob = cp.Problem(cp.Minimize(np.zeros(n)@x),
                      soc_constraints + [A @ x == 0])
    try:
        if (solver == 'GUROBI'): 
            prob.solve(solver=cp.GUROBI)
        elif (solver == 'MOSEK'):
            prob.solve(solver=cp.MOSEK)
        elif (solver == 'ECOS'):
            prob.solve(solver=cp.ECOS)
        elif (solver == 'SCS'):
            prob.solve(solver=cp.SCS)
        else:
            prob.solve()
        xL = x.value/np.linalg.norm(x.value)  # normalize 
        if len(xL) > 0:
            feas = 1 ; xLperp = None
            return xL, xLperp, feas
    except:
        xL = None; feas = -1; 
    
    # Second, for L^\perp \cap K
    prob = cp.Problem(cp.Minimize(np.zeros(n)@x),
                      soc_constraints + [AA @ x == 0])
    try:
        if (solver == 'GUROBI'): 
            prob.solve(solver=cp.GUROBI)
        elif (solver == 'MOSEK'):
            prob.solve(solver=cp.MOSEK)
        elif (solver == 'ECOS'):
            prob.solve(solver=cp.ECOS)
        elif (solver == 'SCS'):
            prob.solve(solver=cp.SCS)
        else:
            prob.solve()
        xLperp = x.value/np.linalg.norm(x.value)  # normalize 
        if len(xLperp) > 0:
            feas = 2 ; xL = None
            return xL, xLperp, feas
    except:
        xLperp = None ; feas = -1
    return xL,xLperp,feas


def PRAvariant(A, AA, K, z0, aggressive = True, RescalingLimit = 50):
    """ Projection and Rescaling Algorithm
     This is nearly identical to PRA.  
     The only difference are the updates of the projection matrices.
     *Inputs:*
     A and AA       : Matrices such that L = null(A) and Lperp = null(AA) 
     K              : Direct product of second-order cones
     z0             : Initial point in K
     aggressive     : Boolean variable to enable agressive rescaling heuristic
     RescalingLimit : Upper bound on the number of rescaling rounds 
    
     *Outputs:*
     feas           : Feasibility status of the problem identified by the algorithm.
                      feas = 1 when found an interior point xL in L \cap K
                      feas = 2 when found an interior point xLperp in L \cap K
                      feas = -1 when reached RescalingLimit without solving either problem
     xL             : Solution found in L\cap K
     xLperp         : Solution found in Lperp \cap K
     k              : Number of rescaling rounds
     Total          : Total number of iterations
    """
    Q,R = np.linalg.qr(AA.T) 
    P = Q@Q.T
    m = AA.shape[0]; n = AA.shape[1]

    QQ, R = np.linalg.qr(A.T)
    PP = QQ@QQ.T 

    # Initialization*
    Total = 0; k = 0; feas = 0 
    zz0 = z0 
    D = np.eye(len(z0)); DD = D 

    while (feas == 0) and (k <= RescalingLimit):   
        # ** Basic procedure phase (smooth perceptron algorithm)
        # Primal
        y, z, totalitrsmooth, feasprimal = basic(K, P, z0, 0.2) 
        Total += totalitrsmooth         
        # Dual 
        yLperp, zLperp, totalitrsmooth,feasdual = basic(K, PP, zz0, 0.2) 
        Total += totalitrsmooth 

        # ** Check if basic procedure succeeded
        if feasprimal:
            feas = 1
            xL = np.linalg.solve(D,y) 
            xL = xL/np.linalg.norm(xL)  # normalize 
            xLperp = xL*0
            return feas,xL,xLperp,k,Total
        if feasdual:
            feas = 2
            xLperp = np.linalg.solve(DD,yLperp)
            xLperp = xLperp/np.linalg.norm(xLperp)  # normalize 
            xL = xLperp*0
            return feas,xL,xLperp,k,Total
    
        # ** Rescaling phase -- primal 
        B = rescale(K,P,z,aggressive)            
        D = B@D        
        # Update the projection matrix after rescaling
        # Q,R = np.linalg.qr(D@AA.T)
        BmI = B - np.eye(n); L,V = np.linalg.eigh(Q.T@(2*BmI+BmI@BmI)@Q) ; 
        R = np.eye(m) - V@np.diag((1+L)**(-0.5)+1)@V.T 
        Q = B@Q@R
        P = Q@Q.T 

        # ** Rescaling phase -- dual
        B = rescale(K,PP,zLperp,aggressive)
        DD = B@DD 
        # Update the projection matrix after rescaling       
        # QQ,R = np.linalg.qr(DD@A.T) 
        BmI = B - np.eye(n); L,V = np.linalg.eigh(QQ.T@(2*BmI+BmI@BmI)@QQ) ; 
        R = np.eye(n-m) - V@np.diag((1+L)**(-0.5)+1)@V.T 
        QQ = B@QQ@R
        PP = QQ@QQ.T
        k = k+1 
    if (k > RescalingLimit):
        print('Could not finish within '+str(RescalingLimit)+' iterations')
        feas = -1 ; xL = 0*z0; xLperp = 0*z0
        return feas,xL,xLperp,k,Total
