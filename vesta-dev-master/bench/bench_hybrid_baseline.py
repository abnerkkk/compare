import os, sys
sys.path.insert(0, os.path.abspath("."))
from vesta import *
from scipy.optimize import *
from line_solver import Exp, Erlang
import time

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# Define Objective Function
def fun(x):
    topology = ServiceTopology("topology")
    cores = max(1, int(round(x[3])))
    host1 = Host(topology, "host1", cores=cores)
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
    print(t)
    return -t

# Run
if __name__ == '__main__':
    cons = ({'type': 'ineq', 'fun': lambda x: 10 - x[0] - x[1]}, {'type': 'ineq', 'fun': lambda x: -(10 - x[0] - x[1])})
    bnds = ((0.001, 10), (0.001, 15), (0,1), (1,2))
    tic = time.perf_counter()
    #res = minimize(fun, (0.001, 0.001), method='SLSQP', bounds=bnds, constraints=cons, tol=1e-6, options={"maxiter": 50, "disp": True})
    #res = minimize(fun, (0.001, 0.001), method='COBYLA', bounds=bnds, constraints=cons, tol=1e-6, options={"maxiter": 50, "disp": True})
    #res = minimize(fun, (0.001, 0.001), method='trust-constr', bounds=bnds, constraints=cons, tol=1e-6, options={"maxiter": 50, "disp": True})
    lc = LinearConstraint([[1, 1, 0, 0]], 10, 10)
    res = differential_evolution(fun, bnds, maxiter=60, popsize=2, constraints=lc, integrality=(0,0,1,1))
    #res = shgo(fun, bnds, constraints=lc)

    print(res)
    toc = time.perf_counter()
    print(f"Time: {toc - tic:0.4f} seconds")
