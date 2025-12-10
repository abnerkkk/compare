import os, sys, time
sys.path.insert(0, os.path.abspath("."))
from vesta import *
from scipy.optimize import *
import time

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# Define Objective Function
def fun(x):
    t = opt_lqn_8(x)
    y = -t
    return y

# Run
if __name__ == '__main__':
    bnds = ((0.001, 10), (0.001, 15), (0.001, 20), (0.001, 25), (1,2), (1,5), (2,5), (2,4))
    tic = time.perf_counter()

    ## evolutionary
    lc = LinearConstraint([[1, 1, 0, 0, 0, 0, 0, 0]], 10, 10)
    res = differential_evolution(fun, bnds, maxiter=200, popsize=2, polish=False, constraints=lc, integrality=(0,0,0,0,1,1,1,1), disp=True)

    print(res)
    print(opt_lqn_8(res.x))
    toc = time.perf_counter()
    print(f"Time: {toc - tic:0.4f} seconds")