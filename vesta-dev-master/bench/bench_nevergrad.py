import os, sys, time
sys.path.insert(0, os.path.abspath("."))
from vesta import *
import nevergrad as ng

def project(x):
    # enforce bounds + (optional) exact sum constraint by projection
    x0 = float(x[0])
    x0 = min(max(x0, 0.0), 10.0)
    x1 = 10.0 - x0
    return (x0, x1)

def fun(x):
    x0, x1 = project(x)
    # if you want soft constraint instead, comment project() and do penalty
    t = opt_lqn_2((x0, x1))
    return -t

if __name__ == '__main__':
    tic = time.perf_counter()

    # 1D search is enough because x1 is determined by x0
    instrum = ng.p.Scalar(lower=0.0, upper=10.0)  # x0
    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=10)

    recommendation = optimizer.minimize(lambda x0: fun((x0, 0.0)), verbosity=2)

    x0 = float(recommendation.value)
    x0, x1 = project((x0, 0.0))
    print("best (x0,x1) =", (x0, x1))
    print("objective =", fun((x0, x1)))

    toc = time.perf_counter()
    print(f"Time: {toc - tic:0.4f} seconds")