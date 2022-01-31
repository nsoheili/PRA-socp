Implementation of a projection and rescaling algorithm for second-order conic feasibility problems.

We provide a Python implementation of the projection and rescaling algorithm for second-order conic   feasibility problems  as described in the manuscript "Implementation of a projection and rescaling algorithm for second-order conic feasibility problems" by J. Pena and N. Soheili.   

The Python code has been organized in two main files: PRAalgorithms.py and PRAtests.py.

The file PRAalgorithms.py contains the code that implements all of the routines required in the projection and rescaling algorithm.  The main class and functions are the following:

class Cone():  A Python class to construct instances of a second-order cone object.  For a vector $dim = [d_1,...d_ell] the command K = Cone(dim) constructs the direct product of second-order cones of dimensions d_1,...,d_ell.

def PRA(A,AA,K,u0,aggressive=True,RescalingLimit=50): Python implementation of the projection and rescaling algorithm.  Assume K is a direct product of second order cones, L:=ker(A) and Lperp:=ker(AA) are complementary linear subspaces in the same Euclidean space, and u0 is a point in int(K) whose eigenvalues add up to one.   The boolean variable  "aggressive" enables the aggressive rescaling heuristic.  The variable  "RescalingLimit" enforces an upper bound on the number of rescaling rounds.  The algorithm returns the output variables: feas, xL, xLperp, k, Total.  The first three outputs are as follows:  

feas = 1  when the algorithm terminates with a point xL in L and int(K) and xLperp = 0, 

feas = 2  when the algorithm terminates with xLperp  in Lperp and int(K) and xL = 0, 

feas = -1 when the algorithm performs "RescalingLimit" rescaling rounds and forcibly terminates with  xL = xLperp = 0.  

The last two outputs k, Total are respectively the number of rescaling rounds and total number of iterations.

The file PRAtests.py contains some auxiliary functions to test the projection and rescaling algorithm on a collection of problem instances.

def experiments(lset,dset,deltaset,limdim,N): Python function to construct a direct product of second cone, generate N random instances, and test the above PRA algorithm.  The N instances are generated for each choice of the parameters ell and (d_1,\dots,d_ell) defining the cone, as well as a conditioning parameter delta, defined via the sets  lset, dset, deltaset.  This algorithm returns the dataframes dsum, dfResult that summarize the numerical results and generates tables and summary plots.

def comparison(rset,n,deltaset,N): Python function to construct a direct product of second cones, and then compare the PRA algorithm with alternative solvers on N random instances.  The N instances are generated for each choice of the parameters ell, (d_1,\dots,d_\ell), and delta defined via the sets lset, dset, deltaset.  This algorithm returns the dataframes compsuccess, compCPU and also largestnorm, smallestminev}  These dataframes summarize the numerical results and generate tables.

The additional files PRAdemo.py and PRAdemo.ipynb  perform experiments via the previous functions and summary tables and plots for a small set of low-dimensional test problems. 
