from vesta import *
import numpy as np
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.evolutionary import *
from opytimizer.optimizers.swarm import *
from opytimizer.spaces import SearchSpace
import time


# Define Objective Function
def fun(x):
    x0 = x[0]
    x1 = x[1]
    t = opt_lqn_2(x)
    rho = 10
    y = -t * (1 + rho * np.exp(-abs(10 - x0 - x1)))
    return y.reshape((-1, 1)).item()

# Run
if __name__ == '__main__':
    tic = time.perf_counter()

    n_agents = 1
    n_variables = 2
    lower_bound = [0.001, 10]
    upper_bound = [0.001, 15]
    space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
    optimizer = DE()
    function = Function(fun)

    opt = Opytimizer(space, optimizer, function)
    opt.start(n_iterations=50)
    toc = time.perf_counter()
    print(f"Time: {toc - tic:0.4f} seconds")
    print(
        f"Best Agent: {opt.space.best_agent.mapped_position} | Fitness: {opt.space.best_agent.fit}"
    )
