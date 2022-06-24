#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRA code -- additional files for testing
Modified on Friday February 4, 2022

@author: javipena
"""
import numpy as np
import time
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter
from PRAalgos import Cone,PRA,PRAsocp,NaiveInstance,ControlledInstance        

def comparison(rset,n,deltaset,N):
    
    """
    Script to perform some comparisons between PRA and other SOCP solvers: GUROBI, MOSEK, ECOS, SCS.

    Parameters
    ----------
    lset : tuple of ell-values
    n: problem dimension -- it should be a multiple of all entries in rset
    deltaset : 4-dimensional tuple of delta-values
    N : Number of random experiments for each ell, d, delta

    Returns
    -------
    compsuccess: number of instances solved successfully for each solver (PRA, GUROBI, MOSEK, ECOS, SCS)
    compCPU: average CPU time among successfully solved instances
    largestnorm: largest norm of Ax for output x over successfully solved instances
    smallestminev: smallest of the minimum eigenvalue of x over successfully solved instances
    """

    solvers = ['PRA','GUROBI','MOSEK','ECOS']
    CPUresults = pd.DataFrame(columns=["PRA",'GUROBI','MOSEK','ECOS',"delta", "r"])
    CPUresultsnet = pd.DataFrame(columns=["PRA",'GUROBI','MOSEK','ECOS',"delta", "r"])
    normresults = pd.DataFrame(columns=["PRA",'GUROBI','MOSEK','ECOS',"delta", "r"])
    minevresults = pd.DataFrame(columns=["PRA",'GUROBI','MOSEK','ECOS',"delta", "r"])
    compsuccess = pd.DataFrame(index = solvers) 
    compCPU = pd.DataFrame(index = solvers) 
    compCPUnet = pd.DataFrame(index = solvers) 
    largestnorm = pd.DataFrame(index = solvers) 
    smallestminev = pd.DataFrame(index = solvers) 
    for delta in deltaset:
        for r in rset:
            d = n/r ; dim = np.ones(r)*d ; dim = dim.astype(int);
            K = Cone(dim) ; n = sum(K.dim) ; 
            m = n/2 ; m = m.astype(int) ;
            cputimes, cputimesnet, normresiduals, mineigenvals = comparePRA(m,K,N,delta,True,50) 
            cpudf = pd.DataFrame(cputimes.T,columns = solvers) ; cpudf[['delta','r']] = [delta,r]
            cpudfnet = pd.DataFrame(cputimesnet.T,columns = solvers) ; cpudfnet[['delta','r']] = [delta,r]
            normsdf = pd.DataFrame(normresiduals.T,columns = solvers) ; normsdf[['delta','r']] = [delta,r]
            minevdf = pd.DataFrame(mineigenvals.T,columns = solvers) ; minevdf[['delta','r']] = [delta,r]

            CPUresults = CPUresults.append(cpudf, ignore_index=True) 
            CPUresultsnet = CPUresultsnet.append(cpudfnet, ignore_index=True) 
            normresults = normresults.append(normsdf, ignore_index=True) 
            minevresults = minevresults.append(minevdf, ignore_index=True) 
            compresults = pd.concat([CPUresults,CPUresultsnet,normresults,minevresults], axis = 0, keys = ['CPU','CPUnet','norm','minev'],names = ['metric'])
            compresults.to_csv('compresults.csv')
            
    # Index by delta and r for convenience
    CPUresults = CPUresults.set_index(['delta','r']) 
    CPUresultsnet = CPUresultsnet.set_index(['delta','r']) 
    normresults = normresults.set_index(['delta','r']) 
    minevresults = minevresults.set_index(['delta','r']) 
    compresults = pd.concat([CPUresults,CPUresultsnet,normresults,minevresults], axis = 0, keys = ['cputime','cputimenet','norm','minev'],names = ['metric'])
    compresults.to_csv('compresults.csv')

    # Compute summaries
    compsuccess = CPUresults.groupby(level=['delta','r']).count().astype(float)/N 
    compCPU = CPUresults.groupby(level=['delta','r']).mean() 
    compCPUnet = CPUresultsnet.groupby(level=['delta','r']).mean() 
    largestnorm = normresults.groupby(level=['delta','r']).max() 
    smallestminev = minevresults.groupby(level=['delta','r']).min() 
    compsummary = pd.concat([compsuccess,compCPU,compCPUnet,largestnorm,smallestminev], axis = 0, keys = ['successavg','cputimeavg','cputimenetavg','largestnorm','smallestminev'],names = ['metric'])
    compsummary.to_csv('compsummary.csv')

    print('\n\n Proportion of instances solved successfully\n') 
    print(compsuccess)
    # Report the net CPU times only
    print('\n\n Average CPU times of instances solved successfully\n') 
    print(compCPUnet)
    # If we also want the gross CPU times run also
    #    print('\n\n Average CPU times of instances solved successfully\n') 
    #    print(compCPU)


    print('\n\n Largest norm (instances solved successfully)\n') 
    print(largestnorm)
    print('\n\n Smallest minimum eigenvalue (instances solved successfully)\n') 
    print(smallestminev)

    # To be consistent with the paper, return the net CPU times only
    return compsuccess, compCPUnet, largestnorm,smallestminev
    # If we also want the gross CPU times run instead
    #    return compsuccess,compCPU, compCPUnet, largestnorm,smallestminev
    
def comparePRA(m,K,N,delta=0.01,aggressive=True,RescalingLimit=50):
    """Script to compare the projection and rescaling algorithm with other SOCP solvers.
    Input parameters are identical to those of TestPRA.

    Parameters
    ----------
     m              : Codimension of the linear subspace L
     K              : Direct product of second-order cones
     N              : Number of random instances to test
     delta          : Parameter to generate random instances with controlled condition
     aggressive     : Boolean variable to enable agressive rescaling heuristic
     RescalingLimit : Upper bound on the number of rescaling rounds    

    Returns(all N-dimensional arrays)
    -------
     cputimes       : total CPU times (python + SOCP solver)
     cputimesnet    : CPU times (SOCP solver only)
     residualnorm   : either norm(A@xL) or norm(AA@xLperp)
     mineigvalue    : either min(eigenvalue(xL)) or min(eigenvalue(xLperp))
    """
    # Initialization
    cpuPRA = np.zeros(N) 
    cpucp = np.zeros((3,N)) 
    cpucpnet = np.zeros((3,N)) 
    normresiduals = np.zeros((4,N)) 
    mineigenvals = np.zeros((4,N)) 

    solvers = ('GUROBI','MOSEK','ECOS')

    # Initialization of the algorithm inputs
    n = sum(K.dim)
    r = K.r
    u0 = np.zeros(n)
    u0[K.sdim] = 1
    u0 = u0/(2*sum(u0))
    lx = np.zeros(r)
    print("Running experiments for N = " + str(N) + ", r = " + str(r) + ", n = "+ str(sum(K.dim)) + ", delta = " + str(delta))
    for i in range(N):
        if (delta >= 1):
            # Generate naive random instance
            A,AA = NaiveInstance(m,sum(K.dim))         
        else:
            # Generate random instances with controlled condition
            # Generate the most interior solution x
            r0 = int(np.round(np.random.rand()*K.r*0.3)) 
            lx = np.hstack((np.random.rand(r0),delta*np.random.rand(r-r0)))
            lx = lx/max(lx) 
            x = np.zeros(n)
            x[K.sdim]=lx
            # This x so far has a very simple structure.  
            # We next bring some blocks closer to the boundary.
            # Build blocks so that the smaller of i-th eigenvalues is smaller than delta*x_i0.
            for j in range(K.r):
                indx = range(K.dim[j])+K.sdim[j]    
                u = np.random.randn(len(indx)-1) 
                x[indx[1:len(indx)]] = u*x[K.sdim[j]]*(1-delta*np.random.rand())/np.linalg.norm(u)    
            # Generate instance L = null(A) such that x is the most interior
            # solution to L\cap\R^n_+
            A,AA = ControlledInstance(m,x,K) 
            llx,ex = K.eigenvalues(x)
        # Run PRA
        stime = time.time()
        feas,xL,xLperp, multi_k, multi_Total = PRA(A, AA, K, u0, aggressive, RescalingLimit)      
        # Record the CPU time, the number of rescaling iterations, the total number of iterations,
        # and the feasibility type returned by multi-directions EPRA 
        if (feas < 0):
            cpuPRA[i] = np.nan ; normresiduals[0,i] = np.nan ; mineigenvals[0,i] = np.nan ;
        else:
            cpuPRA[i] = time.time() - stime  
            if (feas == 1):
                normresiduals[0,i] = np.linalg.norm(A@xL)
                xLeval, xLevec = K.eigenvalues(xL) 
                mineigenvals[0,i] = min(xLeval)
            else:    
                normresiduals[0,i] = np.linalg.norm(AA@xLperp)
                xLeval, xLevec = K.eigenvalues(xLperp) 
                mineigenvals[0,i] = min(xLeval)
        for j in range(3):
            if (feas == 2): # swap the roles of A and AA to favor/speed up the cp version
                AAA = AA ; AA = A ; A = AAA
            stime = time.time()        
            xL,xLperp,feas,socptime = PRAsocp(A,AA,K,solvers[j])
            if (feas<0):
                cpucp[j,i] = np.nan ; normresiduals[j+1,i] = np.nan ; mineigenvals[j+1,i] = np.nan ;
                cpucpnet[j,i] = np.nan
            else:
                cpucp[j,i] = time.time() - stime  ; cpucpnet[j,i] = socptime
                if (feas == 1):
                    normresiduals[j+1,i] = np.linalg.norm(A@xL)
                    xLeval, xLevec = K.eigenvalues(xL) 
                    mineigenvals[j+1,i] = min(xLeval)
                else:    
                    normresiduals[j+1,i] = np.linalg.norm(AA@xLperp)
                    xLeval, xLevec = K.eigenvalues(xLperp) 
                    mineigenvals[j+1,i] = min(xLeval)
                if min(xLeval) < 0:
                    cpucp[j,i] = np.nan ; normresiduals[j+1,i] = np.nan ; mineigenvals[j+1,i] = np.nan ; 
                    cpucpnet[j,i] = np.nan
        cputimes = np.vstack((cpuPRA,cpucp)) ; cputimesnet = np.vstack((cpuPRA,cpucpnet)) ;
    return cputimes, cputimesnet, normresiduals, mineigenvals  

    
def TestPRA(m,K,N,delta=0.01,aggressive=True,RescalingLimit=50):
    """Script to test the projection and rescaling algorithm.

    Parameters
    ----------
     m              : Codimension of the linear subspace L
     K              : Direct product of second-order cones
     N              : Number of random instances to test
     delta          : Parameter to generate random instances with controlled condition
     aggressive     : Boolean variable to enable agressive rescaling heuristic
     RescalingLimit : Upper bound on the number of rescaling rounds    

    Returns(all N-dimensional arrays)
    -------
     cputime        : CPU times
     rescalingiterations : Number of rescaling rounds
     totaliterations: Total number of basic iterations
     feasibility    : Feasibility status 
     condition      : delta condition measure     
    """
    # Initialization
    cputime = np.zeros(N) 
    feasibility = np.zeros(N) 
    totaliterations = np.zeros(N) 
    rescalingiterations = np.zeros(N) 
    condition = np.zeros(N) # records the condition measure of each instance
    # Initialization of the algorithm inputs
    n = sum(K.dim)
    r = K.r
    u0 = np.zeros(n)
    u0[K.sdim] = 1
    u0 = u0/(2*sum(u0))
    lx = np.zeros(r)
    print("Running experiments for N = " + str(N) + ", r = " + str(r) + ", n = "+ str(sum(K.dim)) + ", delta = " + str(delta))
    for i in range(N):
        if (delta >= 1):
            # Generate naive random instance
            A,AA = NaiveInstance(m,sum(K.dim))         
        else:
            # Generate random instances with controlled condition
            # Generate the most interior solution x
            r0 = int(np.round(np.random.rand()*K.r*0.3)) 
            lx = np.hstack((np.random.rand(r0),delta*np.random.rand(r-r0)))
            lx = lx/max(lx) 
            x = np.zeros(n)
            x[K.sdim]=lx
            # This x so far has a very simple structure.  
            # We next bring some blocks closer to the boundary.
            # Build blocks so that the smaller of i-th eigenvalues is smaller than delta*x_i0.
            for j in range(K.r):
                indx = range(K.dim[j])+K.sdim[j]    
                u = np.random.randn(len(indx)-1) 
                x[indx[1:len(indx)]] = u*x[K.sdim[j]]*(1-delta*np.random.rand())/np.linalg.norm(u)    
            # Generate instance L = null(A) such that x is the most interior
            # solution to L\cap\R^n_+
            A,AA = ControlledInstance(m,x,K) 
            llx,ex = K.eigenvalues(x)
            condition[i] = sum(np.log(llx))                
        # Run PRA
        stime = time.time()
        feas,xL,xLperp, multi_k, multi_Total = PRA(A, AA, K, u0, aggressive, RescalingLimit)      
        # Record the CPU time, the number of rescaling iterations, the total number of iterations,
        # and the feasibility type returned by multi-directions EPRA 
        cputime[i] = time.time() - stime                              
        rescalingiterations[i] = multi_k # Record the number of rescaling iterations
        totaliterations[i] = multi_Total # Record the total number of iterations
        feasibility[i] = feas            # Record the feasibility type
    return cputime,rescalingiterations,totaliterations,feasibility,condition

def experiments(rset,dset,deltaset,N,limdim):
    """
    Script to run a batch of experiments of various dimensions and various levels of conditioning

    Parameters
    ----------
    lset : 5-dimensional tuple of ell-values
    dset : 5-dimensional tuple of d-values
    deltaset : 4-dimensional tuple of d-values
    N : Number of random experiments for each ell, d, delta
    limdim : Upper bound on the dimension of each set of experiments

    Returns
    -------
    dsum : summary of experiments
    dfResult : more detailed record of experiments
    """
    
    if (len(rset)!=5) | (len(rset)!=5) | (len(deltaset)!=4):
        print("Wrong dimensions")
        dsum = []
        dfResult = []
        return dsum, dfResult
    
    dfResult = pd.DataFrame(columns=["f","delta", "r", "d", "n", "dim", "rescaling", "totaliter", "cputime", "condition"])

    fixdim = (0,1)
    for f in fixdim:
        for delta in deltaset: 
            for r in rset:
                for d in dset: 
                    if f == 0:
                        dim = np.ndarray(r,dtype='int')
                        dim = 0*dim + d
                        K = Cone(dim)
                        r = len(dim)
                        n = sum(dim)
                        m = int(n/2)
                    else:
                        dim = np.random.randint(2,d+1,r)
                        K = Cone(dim)
                        r = len(dim)
                    n = sum(dim)
                    m = int(n/2)
                    if (n<=limdim):
                        cputime,rescaling,totaliter,feasibility,condition = TestPRA(m,K,N,delta,limdim)
                        batchresults = pd.DataFrame({'rescaling': rescaling, 'totaliter': totaliter, 'cputime': cputime, 'condition': condition})
                        batchresults[['f','delta','r','d','n']] = [f,delta,r,d,n] ; batchresults['dim'] = str(dim)
                        dfResult = dfResult.append(batchresults, ignore_index=True)
                        dfResult.to_csv('PRAresults.csv')
    dfResult = dfResult.set_index(['f','delta','r'])
    dfResult.to_csv('PRAresults.csv')
    dsum = dfResult.groupby(level = ['f','delta','r']).mean()
    dsum.to_csv('PRAsummary.csv')
    print('\n\n Summary of experiments\n') 
    print(dsum)

    create_plots_aggregated_scatter(dfResult,"rescaling",rset,deltaset)    
    create_plots_aggregated_scatter(dfResult,"totaliter",rset,deltaset)    
    create_plots_scatter(dfResult,"rescaling",rset,dset,deltaset) 
    create_plots_scatter(dfResult,"totaliter",rset,dset,deltaset)

    return dsum,dfResult


# Plotting functions
def create_plots_scatter(data,field,rvalues,dvalues,deltaset):

    # ****************************************
    # Plotting parameters  MOVE LATER
    fontsize = 18 ; markersize = 10 ; labelpad = 20; alpha = 0.7
    
    deltacolor = {deltaset[3]:'b',deltaset[2]:'orange',deltaset[1]:'g',deltaset[0]:'r'}
    
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Small delta',
                          markerfacecolor='blue', markersize=markersize),
                       Line2D([0], [0], marker='o', color='w', label='Medium delta',
                          markerfacecolor='green', markersize=markersize),
                       Line2D([0], [0], marker='o', color='w', label='Large delta',
                          markerfacecolor='orange', markersize=markersize),
                       Line2D([0], [0], marker='o', color='w', label='Naive',
                          markerfacecolor='red', markersize=markersize)]

            
    plt.figure(0, tight_layout = False, figsize = (16,12))
    k = 0 # Indix k determines the location of the plot in a 5 * 5 grid
    flagcontinue = False # flagcontinue determines where the plot location in the grid should be skipped or not
     
    # plot only fixed dimension results
    data = data.loc[0]
    allfielddata = data.reset_index().set_index(['r','d','delta'])[field]
    for i in range(5):
        if k == 17:
            break
        for j in range(5):
            if k == 17:
                break
            if k == 9 or k == 13 or k == 16:
                if flagcontinue == False:
                    flagcontinue = True
                    break
                else:
                    flagcontinue = False
                    
            if k == 0 or k == 6 or k == 9 or k == 13 or k == 16:
                ax0 = plt.subplot2grid((5,5), (i,j))
                axes = ax0
            else:
                axes = plt.subplot2grid((5,5), (i,j), sharex = ax0)
            
                    
            if k == 0 or k ==5 or k == 9 or k == 13 or k == 16: 
                axes.set_ylabel('l = %i' %rvalues[i], fontsize = fontsize, labelpad = labelpad - 10)
                    
            if k <= 4:
                axes.set_title('d = %i' %dvalues[j], fontsize = fontsize, pad = labelpad)
                        
            fielddata = allfielddata[rvalues[i]][dvalues[j]]
            for delta in deltaset:
                fieldvals = fielddata[delta].values
                plt.scatter(range(len(fieldvals)), fieldvals, alpha = alpha, label = 'delta = %1.3f' %delta,color = deltacolor[delta])
                    
            axes.yaxis.set_major_locator(MaxNLocator(6))

            plt.setp(axes.get_xticklabels(), visible=False)
            plt.subplots_adjust(wspace = 0.1)
            axes.label_outer()
            k += 1
            
    # Building an artificial plot around the subplots for adding legend and the labels               
    plt.figure(0).add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Number of "+field+" iterations", fontsize = fontsize, labelpad=labelpad+ 20)

    plt.figure(0).legend(handles=legend_elements,scatterpoints=1, loc='lower center', ncol=4,fontsize=fontsize + 2)               
    # Saving the plot
    # plt.savefig('RescalingScatter.pdf', dpi=300, bbox_inches='tight')            
    plt.show()
    
def create_plots_aggregated_scatter(data,field,rvalues,deltaset):

    # ****************************************
    # Plotting parameters  MOVE LATER
    fontsize = 18 ; markersize = 10 ; labelpad = 20; alpha = 0.7
    
    deltacolor = {deltaset[3]:'b',deltaset[2]:'orange',deltaset[1]:'g',deltaset[0]:'r'}
    
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Small delta',
                          markerfacecolor='blue', markersize=markersize),
                       Line2D([0], [0], marker='o', color='w', label='Medium delta',
                          markerfacecolor='green', markersize=markersize),
                       Line2D([0], [0], marker='o', color='w', label='Large delta',
                          markerfacecolor='orange', markersize=markersize),
                       Line2D([0], [0], marker='o', color='w', label='Naive',
                          markerfacecolor='red', markersize=markersize)]
      
    allfielddata = data.reset_index().set_index(['r','delta'])[field]
    plt.figure(6, tight_layout = False, figsize = (16,8))
    
    maxv = np.zeros(len(rvalues)+1)
    k = 1 
    for j in range(5):
        if j == 0:
            ax0 = plt.subplot2grid((1,5), (0,j))
            axes = ax0
        else:
            axes = plt.subplot2grid((1,5), (0,j), sharex = ax0, sharey = ax0)
        axes.set_title('l = %i' %rvalues[j], pad = labelpad, fontsize = fontsize)
        fielddata = allfielddata[rvalues[j]]
        for delta in deltaset:
            fieldvals = fielddata[delta].values
            plt.scatter(range(len(fieldvals)), np.power(fieldvals,0.5), color = deltacolor[delta], alpha = alpha)
            maxv[k] = max(maxv[k], max(fieldvals))            
        k += 1
        maxdel = np.zeros(len(deltaset) + 1)
        i = 1
        for delv in deltaset:
            p = fielddata[delta]
            for m in p.index:
                maxdel[i] = max(maxdel[i], max(p[m]))
            i += 1        
        if field == "rescaling":
            lim = maxdel
        else:
            lim = maxv                     
        y_formatter = FixedFormatter(lim) 
        y_locator = FixedLocator(np.power(lim, 0.5))
        axes.yaxis.set_major_formatter(y_formatter)
        axes.yaxis.set_major_locator(y_locator)   
        plt.setp(axes.get_xticklabels(), visible=False)
        plt.subplots_adjust(wspace = 0.1)
        axes.label_outer()                                      
    plt.figure(6).add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Number of "+field+" iterations", fontsize = fontsize, labelpad=labelpad+10)
    plt.figure(6).legend(handles=legend_elements,scatterpoints=1, loc='lower center', ncol=4,fontsize=fontsize + 2)       
    plt.show()
    
