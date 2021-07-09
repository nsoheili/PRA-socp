#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 09:10:45 2021

@author: javipena
"""

from PRAtests import experiments

""" 
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
dsum,dfResult = experiments(lset,dset,deltaset,N,limdim)