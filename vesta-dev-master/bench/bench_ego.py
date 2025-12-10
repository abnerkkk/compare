import numpy as np
import time
from vesta import *
from smt.applications import EGO
from smt.surrogate_models import KRG, KPLS
from smt.utils.design_space import *
from smt.applications.mixed_integer import MixedIntegerContext
from smt.surrogate_models import MixIntKernelType
import matplotlib.pyplot as plt

def fun(X):
    y = np.zeros(len(X))
    for j in range(len(X)):
        x0 = X[j, 0]
        x1 = X[j, 1]
        t = opt_lqn_2((x0, x1))
        rho = 10
        y[j] = t * (1 + rho * np.exp(-abs(10 - x0 - x1)))
    return y.reshape((-1, 1))

print("Sampling...")
tic = time.time()
n_iter = 8
seed = 3  # for reproducibility
design_space = DesignSpace(
    [
        FloatVariable(0, 10),
        FloatVariable(0, 15),
    ],
    seed=seed,
)
mixint = MixedIntegerContext(design_space)
n_doe = 8 * len(design_space.design_variables)
sampling = mixint.build_sampling_method(random_state=seed)
xdoe = sampling(n_doe)
ydoe = fun(xdoe)
toc = time.time()
print('Sampling time: %0.4f s' % (toc - tic))

print("Optimizing...")
tic = time.time()
criterion = "SBO"  # 'EI' or 'SBO' or 'LCB'
qEI = "KBRand"
sm = KPLS(theta0=[1e-2],
          design_space=design_space,
          corr="pow_exp",
          print_global=False,
          )
ego = EGO(
    n_iter=n_iter,
    criterion=criterion,
    xdoe=xdoe,
    ydoe=ydoe,
    surrogate=sm,
    qEI=qEI,
    n_parallel=1,
    random_state=seed,
)
x_opt, y_opt, _, _, y_data = ego.optimize(fun=fun)
print("Minimum in x={} with f(x)={:.4f}".format(x_opt, float(y_opt)))
toc = time.time()
print('Optimization time: %0.4f s' % (toc - tic))

print(opt_lqn_2(x_opt))
# 4.998e+00  5.002e+00
