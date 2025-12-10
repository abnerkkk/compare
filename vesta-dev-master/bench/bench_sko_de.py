from vesta import *
from sko.PSO import PSO
from sko.DE import DE
from sko.SA import SA
import time


# Define Objective Function
def fun(x):
    t = opt_lqn_2(x)
    y = -t
    return y


constraint_eq = [
    lambda x: 10 - x[0] - x[1],
    lambda x: 4 - x[0] - x[1]
]

# Run
if __name__ == '__main__':
    tic = time.perf_counter()
    opt = DE(func=fun, n_dim=2, size_pop=4, max_iter=2, lb=[0.001, 0.001], ub=[10, 15], constraint_eq=constraint_eq)
    opt.run()
    print('best_x is ', opt.best_x, 'best_y is', opt.best_y)
    toc = time.perf_counter()
    print(f"Time: {toc - tic:0.4f} seconds")
    print(fun((5,5)))