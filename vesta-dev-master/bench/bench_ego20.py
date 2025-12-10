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
        x2 = X[j, 2]
        x3 = X[j, 3]
        x4 = X[j, 4]
        x5 = X[j, 5]
        x6 = X[j, 6]
        x7 = X[j, 7]
        x8 = X[j, 8]
        x9 = X[j, 9]
        x10 = X[j, 10]
        x11 = X[j, 11]
        x12 = X[j, 12]
        x13 = X[j, 13]
        x14 = X[j, 14]
        x15 = X[j, 15]
        x16 = X[j, 16]
        x17 = X[j, 17]
        x18 = X[j, 18]
        x19 = X[j, 19]
        t = opt_lqn_20((x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19))
        rho = 10
        y[j] = -t * (1 + rho * np.exp(-abs(10 - x0 - x1)))
    return y.reshape((-1, 1))


print("Sampling...")
tic = time.time()
n_iter = 4
seed = 23000  # for reproducibility
design_space = DesignSpace(
    [
        FloatVariable(0.0001, 10),
        FloatVariable(0.0001, 10),
        FloatVariable(0.0001, 10),
        FloatVariable(0.0001, 10),
        FloatVariable(0.0001, 10),
        FloatVariable(0.0001, 10),
        FloatVariable(0.0001, 10),
        FloatVariable(0.0001, 10),
        FloatVariable(0.0001, 10),
        FloatVariable(0.0001, 10),
        IntegerVariable(1, 2),
        IntegerVariable(1, 2),
        IntegerVariable(1, 3),
        IntegerVariable(1, 4),
        IntegerVariable(1, 5),
        IntegerVariable(1, 6),
        IntegerVariable(1, 7),
        IntegerVariable(1, 8),
        IntegerVariable(1, 9),
        IntegerVariable(1, 10),
    ],
    seed=seed,
)
mixint = MixedIntegerContext(design_space)
n_doe = 3 * len(design_space.design_variables)
sampling = mixint.build_sampling_method(random_state=seed)
xdoe = sampling(n_doe)
ydoe = fun(xdoe)
toc = time.time()
print('Sampling time: %0.4f s' % (toc - tic))

print("Optimizing...")
tic = time.time()
criterion = "SBO"  # 'EI' or 'SBO' or 'LCB'
qEI = "KBRand"
sm = KRG(theta0=[1e-2],
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
    n_parallel=2,
    random_state=seed,
)
x_opt, y_opt, _, _, y_data = ego.optimize(fun=fun)
print("Minimum in x={} with f(x)={:.4f}".format(x_opt, float(y_opt)))
toc = time.time()
print('Optimization time: %0.4f s' % (toc - tic))

print(opt_lqn_20(x_opt))
# 4.998e+00  5.002e+00
