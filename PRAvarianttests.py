#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRA code -- additional files for testing
Modified on Tuesday June 30, 2021

@author: javipena
"""
import numpy as np
import time
import pandas as pd
import statistics
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter
from PRAalgos import Cone,PRAvariant,PRAcpversion,NaiveInstance,ControlledInstance        

from tabulate import tabulate

def comparison(rset,n,deltaset,N):
    """
    Perform some comparisons between PRA and the other solvers: GUROBI, MOSEK, ECOS, SCS
    """
    solvers = ['PRA','GUROBI','MOSEK','ECOS','SCS'] 
    compcount = pd.DataFrame(index = solvers) 
    compmean = pd.DataFrame(index = solvers) 
    compresults = pd.DataFrame(columns=["delta", "r", "n", "m", "PRA",'GUROBI','MOSEK','ECOS','SCS'])
    normresults = pd.DataFrame(columns=["delta", "r", "n", "m", "PRA",'GUROBI','MOSEK','ECOS','SCS'])
    minevresults = pd.DataFrame(columns=["delta", "r", "n", "m", "PRA",'GUROBI','MOSEK','ECOS','SCS'])
    normavg = pd.DataFrame(index = solvers) 
    minevavg = pd.DataFrame(index = solvers) 
    for delta in deltaset:
        for r in rset:
            d = n/r ; dim = np.ones(r)*d ; dim = dim.astype(int);
            K = Cone(dim) ; n = sum(K.dim) ; 
            m = n/2 ; m = m.astype(int) ;
            cputimes, normresiduals, mineigenvals = comparePRA(m,K,N,delta,True,50) 
            cpudf = pd.DataFrame(cputimes,index = solvers)
            normsdf = pd.DataFrame(normresiduals,index = solvers)
            minevdf = pd.DataFrame(mineigenvals,index = solvers)
            """ Execute these commented lines if we want to save each set of experiments separately
            inputarg = delta,r, n, m
            filename = 'Comparison{0}.csv'.format(str(inputarg))
            cpudf.to_csv(filename)"""            
            results = {'delta':delta,'r':r,'n':n,'m':m,'PRA':cpudf.loc['PRA'].values,'GUROBI':cpudf.loc['GUROBI'].values,'MOSEK':cpudf.loc['MOSEK'].values,'ECOS':cpudf.loc['ECOS'].values,'SCS':cpudf.loc['SCS'].values}
            compresults = compresults.append(results, ignore_index=True)
            meanseries = cpudf.mean(axis=1) 
            meanseries.name = '('+str(delta)+','+str(r)+','+str(n)+')'
            countseries = cpudf.count(axis=1) 
            countseries.name = '('+str(delta)+','+str(r)+','+str(n)+')'
            compcount = pd.concat([compcount,countseries],axis=1)            
            compmean = pd.concat([compmean,meanseries],axis=1)
            compmean.to_csv('compmeantimes.csv')
            compcount.to_csv('compsucccount.csv')
            compresults.to_csv('compresults.csv')

            results = {'delta':delta,'r':r,'n':n,'m':m,'PRA':normsdf.loc['PRA'].values,'GUROBI':normsdf.loc['GUROBI'].values,'MOSEK':normsdf.loc['MOSEK'].values,'ECOS':normsdf.loc['ECOS'].values,'SCS':normsdf.loc['SCS'].values}
            normresults = normresults.append(results, ignore_index=True)
            results = {'delta':delta,'r':r,'n':n,'m':m,'PRA':minevdf.loc['PRA'].values,'GUROBI':minevdf.loc['GUROBI'].values,'MOSEK':minevdf.loc['MOSEK'].values,'ECOS':minevdf.loc['ECOS'].values,'SCS':minevdf.loc['SCS'].values}
            minevresults = minevresults.append(results, ignore_index=True)

            meanseries = normsdf.mean(axis=1) 
            meanseries.name = '('+str(delta)+','+str(r)+','+str(n)+')'
            normavg = pd.concat([normavg,meanseries],axis=1)
            meanseries = minevdf.mean(axis=1) 
            meanseries.name = '('+str(delta)+','+str(r)+','+str(n)+')'
            minevavg = pd.concat([minevavg,meanseries],axis=1)

            compmean.to_csv('compmeantimes.csv')
            compcount.to_csv('compsucccount.csv')
            compresults.to_csv('compresults.csv')

            normavg.to_csv('normavg.csv')
            minevavg.to_csv('minevavg.csv')
            normresults.to_csv('normresults.csv')
            minevresults.to_csv('minevresults.csv')


    return compcount,compmean,normavg,minevavg,compresults,normresults,minevresults
    
def comparePRA(m,K,N,delta=0.01,aggressive=True,RescalingLimit=50):
    """Script to compare the projection and rescaling algorithm with a cvxpy version.
    input parameters are identical to those of TestPRA
    output returns focus on cpu times only

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
     cputimes       : CPU times for all solvers (PRA, GUROBI, MOSEK, ECOS, SCS)
     residualnorm   : either norm(A@xL) or norm(AA@xLperp)
     mineigvalue    : either min(eigenvalue(xL)) or min(eigenvalue(xLperp))
    """
    # Initialization
    cpuPRA = np.zeros(N) 
    cpucp = np.zeros((4,N)) 
    normresiduals = np.zeros((5,N)) 
    mineigenvals = np.zeros((5,N)) 

    solvers = ('GUROBI','MOSEK','ECOS','SCS')

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
        feas,xL,xLperp, multi_k, multi_Total = PRAvariant(A, AA, K, u0, aggressive, RescalingLimit)      
        # Record the CPU time, the number of rescaling iterations, the total number of iterations,
        # and the feasibility type returned by multi-directions EPRA 
        if (feas < 0):
            cpuPRA[i] = np.nan 
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
        for j in range(4):
            if (feas == 2): # swap the roles of A and AA to favor/speed up the cp version
                AAA = AA ; AA = A ; A = AAA
            stime = time.time()        
            xL,xLperp,feas = PRAcpversion(A,AA,K,solvers[j])
            if (feas<0):
                cpucp[j,i] = np.nan
            else:
                cpucp[j,i] = time.time() - stime  
                if (feas == 1):
                    normresiduals[j+1,i] = np.linalg.norm(A@xL)
                    xLeval, xLevec = K.eigenvalues(xL) 
                    mineigenvals[j+1,i] = min(xLeval)
                else:    
                    normresiduals[j+1,i] = np.linalg.norm(AA@xLperp)
                    xLeval, xLevec = K.eigenvalues(xLperp) 
                    mineigenvals[j+1,i] = min(xLeval)
        cputimes = np.vstack((cpuPRA,cpucp))  
    return cputimes, normresiduals, mineigenvals  

    
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
    
    dfResult = pd.DataFrame(columns=["f","delta", "r", "d", "dim", "rescaling", "totaliter", "cputime", "condition", "failure"])
    dsum = pd.DataFrame(columns=["f","delta", "r", "d", "n", "rescaling", "totaliter", "cputime", "condition"])
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
                        results = {'f': f, 'delta': delta, 'r': r,'d': d, 'dim': dim, 'rescaling': rescaling, 'totaliter': totaliter, 'cputime': cputime, 'condition': condition}
                        summary = {'f': f, 'delta': delta, 'r': r,'d':d, 'n': n, 'rescaling': statistics.mean(rescaling), 'totaliter': statistics.mean(totaliter), 'cputime': statistics.mean(cputime), 'condition': statistics.mean(condition)}
                        """ Execute these commented lines if we want to save each set of experiments separately
                        inputarg = f,delta,r, n, m
                        dataresults = pd.DataFrame(results)
                        filename = 'Controlled{0}.csv'.format(str(inputarg))
                        dataresults.to_csv(filename) """
                        dsum = dsum.append(summary, ignore_index=True)
                        dfResult = dfResult.append(results, ignore_index=True)
                        dsum.to_csv('summary.csv')
                        dfResult.to_csv('results.csv')
    AggregatedTables(dsum, "rescaling",rset, deltaset)
    AggregatedTables(dsum, "totaliter", rset, deltaset)
    Tables(dsum, "rescaling", dset, rset, deltaset)
    Tables(dsum, "totaliter", dset, rset, deltaset)
    Tables(dsum, "cputime", dset, rset, deltaset)
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
                        
            fielddata = data[(data["f"] == 0) & (data["r"] == rvalues[i]) & (data["d"] == dvalues[j])][field]                    
            nRows = fielddata.index                                        
            for m in nRows:
                delta = data.loc[m,"delta"]
                plt.scatter(range(len(fielddata[m])), fielddata[m], alpha = alpha, label = 'delta = %1.3f' %data.loc[j, "delta"],color = deltacolor[delta])
                    
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
        fielddata = data[data["r"]==rvalues[j]][field]
        nRows = fielddata.index            
        for m in nRows:
            delta = data.loc[m,"delta"]
            plt.scatter(range(len(fielddata[m])), np.power(fielddata[m],0.5), color = deltacolor[delta], alpha = alpha)
            maxv[k] = max(maxv[k], max(fielddata[m]))            
        k += 1
        maxdel = np.zeros(len(deltaset) + 1)
        i = 1
        for delv in deltaset:
            p = data[(data["r"]== rvalues[j]) & (data["delta"] == delv)]["rescaling"]
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
    
    
# Table functions    
 
def Tables(dsum, field, dset, rset, deltaset):
    titles = {"rescaling": "Number of Rescaling Steps","totaliter":"Number of Total Basic Iterations","cputime":"CPU time"}
    print("\n Average "+titles[field]+" for each ell, d, and delta (fixed dimensions only)\n")
    for delta in deltaset:
        print("delta = " + str(delta))
        table = [''] * (len(rset))        
        for i in range(len(rset)):
            df = [''] * (len(dset))
            for j in range(len(dset)): 
                df[j] = (dsum[(dsum["f"] == 0) & (dsum["delta"] == delta) & (dsum["r"] == rset[i]) & (dsum["d"] == dset[j])][field]).to_string(index=False)
                if df[j] == 'Series([], )': df[j] = ''
            df.insert(0, 'l = '+str(rset[i]))
            table[i] = df
        headers = [''] * (len(dset))
        for i in range(0,len(dset)):
            headers[i] = 'd = '+str(dset[i])
        print(tabulate(table, headers, tablefmt="fancy_grid", colalign=("center")))


def AggregatedTables(dsum, field, rset, deltaset):
    titles = {"rescaling": "Number of Rescaling Steps","totaliter":"Number of Total Basic Iterations","cputime":"CPU time"}
    print("\n Average "+titles[field]+" for each ell and delta (fixed and mixed dimensions for all d values)\n")
    table = [''] * (len(rset))        
    for i in range(len(rset)):
        df = [''] * len(deltaset)
        for j in range(len(deltaset)): 
            data0 = dsum[(dsum["f"]==0) & (dsum["delta"] == deltaset[j]) & (dsum["r"] == rset[i])][field]
            data0 = np.round(statistics.mean(data0),2)
            data1 = dsum[(dsum["f"]==1) & (dsum["delta"] == deltaset[j]) & (dsum["r"] == rset[i])][field]
            data1 = np.round(statistics.mean(data1),2)
            df[j] = '('+str(data0)+','+str(data1)+')'
            if df[j] == 'Series([], )': df[j] = ''
        df.insert(0, 'l = '+str(rset[i]))
        table[i] = df
    headers = [''] * (len(deltaset))
    for i in range(len(deltaset)):
        headers[i] = 'delta = '+str(deltaset[i])
    print(tabulate(table,headers, tablefmt="fancy_grid"))


