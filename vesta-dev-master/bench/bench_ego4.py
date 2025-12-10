import numpy as np
import time
import os, sys
sys.path.insert(0, os.path.abspath("."))
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
        t = opt_lqn_4((x0, x1, x2, x3))
        rho = 10
        y[j] = -t * (1 + rho * np.exp(-abs(10 - x0 - x1)))
    return y.reshape((-1, 1))


print("Sampling...")
tic = time.time()
n_iter = 8
seed = 3  # for reproducibility
design_space = DesignSpace(
    [
        FloatVariable(0, 10),# x0：service1 的服务时间分布参数（Exp(0.001 + x0)）
        FloatVariable(0, 15),# x1：service2 的服务时间分布参数（Erlang(0.001 + x1, 2)）
        IntegerVariable(0, 1),# x2 ∈ {0,1}：部署开关（swap）/x2=1：container1→host1，container2→host2 / x2=0：container2→host1，container1→host2（交换部署）
        IntegerVariable(1, 2),# x3 ∈ {1,2}：host1 的核数/并行度（cores）
    ],
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
    n_parallel=1,
    random_state=seed,
)
x_opt, y_opt, _, _, y_data = ego.optimize(fun=fun)
print("Minimum in x={} with f(x)={:.4f}".format(x_opt, float(y_opt)))
toc = time.time()
print('Optimization time: %0.4f s' % (toc - tic))

print(opt_lqn_4(x_opt))
# 4.998e+00  5.002e+00
