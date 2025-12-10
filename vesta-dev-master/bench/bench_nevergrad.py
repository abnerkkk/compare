import warnings
from vesta import *
import nevergrad as ng
import time

# Does not work for some reason due to topology.solve()

# Define Objective Function
def fun(x):
    t = opt_lqn_2(x)
    y = -t #(-t - np.exp(-np.abs(10 - sum(x))))
    return y

# Run
if __name__ == '__main__':
    tic = time.perf_counter()
    # optimization on x as an array of shape (2,)
    #instrum = ng.p.Instrumentation(
    #    ng.p.Array(shape=(2,)).set_bounds(lower=0.0, upper=15),
    #)
    # Initialize NGOpt optimizer
    optimizer = ng.optimizers.NGOpt(parametrization=2, budget=10)

    #optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=50)
    optimizer.parametrization.register_cheap_constraint(lambda x: sum(x) == 10)
    optimizer.parametrization.register_cheap_constraint(lambda x: x[0] >= 0)
    optimizer.parametrization.register_cheap_constraint(lambda x: x[1] >= 0)
    optimizer.parametrization.register_cheap_constraint(lambda x: x[0] <= 10)
    optimizer.parametrization.register_cheap_constraint(lambda x: x[1] <= 15)

    recommendation = optimizer.minimize(fun, verbosity=2)
    print(recommendation.value)

    #optimizer = ng.optimizers.OnePlusOne(parametrization=instrum, budget=50)
    #optimizer = ng.optimizers.ScrHammersleySearchPlusMiddlePoint(parametrization=instrum, budget=50)
    #optimizer = ng.optimizers.PSO(parametrization=instrum, budget=50)
    #optimizer = ng.optimizers.PortfolioDiscreteOnePlusOne(parametrization=instrum, budget=50)
    #optimizer = ng.optimizers.CMA(parametrization=3, budget=100)
    #recommendation = optimizer.minimize(fun)
    #print(recommendation.value)
    x0=recommendation.value[0][0][0]
    x1=recommendation.value[0][0][1]
    print(fun((x0,x1)))
    toc = time.perf_counter()
    print(f"Time: {toc - tic:0.4f} seconds")
