from vesta import *
from scipy.optimize import *
import time

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# Define Objective Function
def fun(x):
    t = opt_lqn_2(x)
    y = -t
    return y


# Run
if __name__ == '__main__':
    cons = ({'type': 'ineq', 'fun': lambda x: 10 - x[0] - x[1]}, {'type': 'ineq', 'fun': lambda x: -(10 - x[0] - x[1])})
    bnds = ((0.001, 10), (0.001, 15))
    tic = time.perf_counter()

    ## evolutionary
    lc = LinearConstraint([[1, 1]], 10, 10)
    res = differential_evolution(fun, bnds, maxiter=100, popsize=15, constraints=lc, integrality=(0,0), disp=True)

    print(res)
    toc = time.perf_counter()
    print(f"Time: {toc - tic:0.4f} seconds")