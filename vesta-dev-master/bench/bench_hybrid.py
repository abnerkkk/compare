import os, sys
sys.path.insert(0, os.path.abspath("."))
from vesta import *
from scipy.optimize import *
from line_solver import Exp, Erlang
import time
import math
import numpy as np

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# Define Objective Function
def inner_opt(x):
    topology = ServiceTopology("topology")
    host1 = Host(topology, "host1", cores=x[3])
    host2 = Host(topology, "host2")
    container1 = Container(topology, "container1")
    container2 = Container(topology, "container2")
    service1 = Service(topology, container1, "service1", Exp(0.001 + x[0]))
    service2 = Service(topology, container2, "service2", Erlang(0.001 + x[1],2))
    users = SynchCaller(topology, "serviceUsers", 1, Exp(1))
    graph = users.init_dag()
    graph.add_edges_from([(service1, service2)])
    users.set_dag(graph)
    if x[2]==1:
        container1.on(host1)
        container2.on(host2)
    else:
        container2.on(host1)
        container1.on(host2)
    topology.solve()
    t = users.tput()
    if not math.isfinite(t) or t is None:
        t = 10**6
    print(t)
    return -t

def outer_opt(y):
    cons = ({'type': 'ineq', 'fun': lambda x: 10 - x[0] - x[1]}, {'type': 'ineq', 'fun': lambda x: -(10 - x[0] - x[1])})
    bnds = ((0.001, 10), (0.001, 15))
    x0 = [np.random.uniform(low, high) for low, high in bnds]
    res = minimize(lambda x: inner_opt((x[0],x[1],y[0],y[1])), x0, method='SLSQP', bounds=bnds, constraints=cons, tol=1e-2, options={"maxiter": 30, "disp": False})
    print(res)
    #res = minimize(lambda x: inner_opt((x[0],x[1],y[0],y[1])), x0, method='COBYLA', bounds=bnds, constraints=cons, tol=1e-2, options={"maxiter": 30, "disp": False})
    return res.fun

# Run
if __name__ == '__main__':
    bnds = ((0,1), (1,2))
    tic = time.perf_counter()
    res = differential_evolution(outer_opt, bnds, maxiter=10, popsize=2, integrality=(1,1), disp=True)
    print(res)
    toc = time.perf_counter()
    print(f"Time: {toc - tic:0.4f} seconds")
