#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 09:10:45 2021

@author: javipena
"""

import numpy as np
from PRAalgos import Cone,PRA,PRAsocp,NaiveInstance,ControlledInstance
from PRAtests import experiments,comparePRA,comparison

"""
First some simple examples
"""
r = 10; d = 10 ; dim = np.ones(r)*d ; dim = dim.astype(int);
K = Cone(dim) ; n = sum(K.dim) ; 
m = n/2 ; m = m.astype(int) ;
A,AA = NaiveInstance(m,sum(K.dim)) ; 
u0 = np.zeros(n) ; u0[K.sdim] = 1 ; u0 = u0/(2*sum(u0))

# Run PRA algorithm and report results

feas,xL,xLperp,k,Total = PRA(A, AA, K, u0)
print('number of rescalings steps = ', k, ',  number of iterations = ',Total)
if (feas == 1): 
    print('found solution in L \cap K') 
    print('norm(xL) = ',str(np.linalg.norm(xL)))    
    print('norm(A@xL) = ', str(np.linalg.norm(A@xL)))
elif (feas == 2): 
    print('found solution in Lperp \cap K') 
    print('norm(xLperp) = ',str(np.linalg.norm(xLperp)))    
    print('norm(AA@xLperp) = ', str(np.linalg.norm(AA@xLperp)))
else:
    print('PRA failed')
    
# Compare with cvxpy version
xL,xLperp,feas,socptime = PRAsocp(A,AA,K,solver = 'MOSEK')
if (feas == 1): 
    print('found solution in L \cap K') 
    print('norm(xL) = ',str(np.linalg.norm(xL)))
    print('norm(A@xL) = ', str(np.linalg.norm(A@xL)))
elif (feas == 2): 
    print('found solution in Lperp \cap K') 
    print('norm(xLperp) = ',str(np.linalg.norm(xLperp)))
    print('norm(AA@xLperp) = ', str(np.linalg.norm(AA@xLperp)))
else:
    print('PRAcpversion failed') 



""" 
Now run experiments like in the paper.
For our experiments len(rset) = len(dset) = 5 and both rset and dset are in increasing order. 
On the other hand, len(deltaset) = 4 and deltaset is in decreasing order.
That way the easier experiments are performed first in case the set of 
experiments is terminated prematurely.
"""    

# Original values
#N = 100
#rset = (10, 20, 50, 100, 500)
#dset = (10, 20, 50, 100, 500)
#deltaset = (1.000, 0.001, 0.010, 0.100)
#limdim = 10000

# Toy example
N = 5
lset = (3, 5, 9, 10, 12)
dset = lset
deltaset = (1.000, 0.500, 0.20, 0.1)
limdim = 200
#dsum,dfResult = experiments(lset,dset,deltaset,N,limdim)

""" 
Run some more experiments to compare PRA with GUROBI, ECOS on controlled instances
"""
deltaset = (1,0.1,0.01,0.001) ; N = 100
#rset = (5,10,20) ; n = 100
rset = (10,20,50,100) ; n = 1000

# Original values
#deltaset = (1,0.1,0.01,0.001) ; N = 100
#rset = (10,20,50,100) ; n = 1000

compsuccess,compCPU,compCPUnet,largestnorm,smallestminev = comparison(rset,n,deltaset,N)
