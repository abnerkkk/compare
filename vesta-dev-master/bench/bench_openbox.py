import os, sys, time
sys.path.insert(0, os.path.abspath("."))
from vesta import *
from openbox import Optimizer, space as sp
import time
import numpy as np

# Define Objective Function
def fun(config):
    x1, x2 = config['x1'], config['x2']
    t = opt_lqn_2((x1, x2))
    c1 = np.exp(abs(10 - x1 - x2))
    result = dict()
    result['objectives'] = [-t]
    result['constraints'] = [c1]
    return result

# Run
if __name__ == '__main__':
    tic = time.perf_counter()
    # Define Search Space
    space = sp.Space()
    x1 = sp.Real("x1", 0.001, 10, default_value=0.001)
    x2 = sp.Real("x2", 0.001, 15, default_value=0.001)
    space.add_variables([x1, x2])

    opt = Optimizer(fun, space, num_constraints=1, max_runs=50, task_id='quick_start')
    history = opt.run()
    toc = time.perf_counter()
    print(history)
    print(f"Time: {toc - tic:0.4f} seconds")
