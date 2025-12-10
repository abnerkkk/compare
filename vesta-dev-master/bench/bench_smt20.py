import os
import sys

# IMPORTANT: Set JAVA_HOME to Java 25 before JVM starts
java_home = r'C:\Program Files\Microsoft\jdk-25.0.1.8-hotspot'
if os.path.exists(java_home):
    os.environ['JAVA_HOME'] = java_home
    os.environ['PATH'] = os.path.join(java_home, 'bin') + os.pathsep + os.environ.get('PATH', '')
    print(f"Using Java from: {java_home}")
else:
    print(f"WARNING: Java not found at {java_home}")

# IMPORTANT: Set PATH before JVM starts
lqns_path = r'C:\Program Files (x86)\LQN Solvers'
os.environ['PATH'] = lqns_path + os.pathsep + os.environ.get('PATH', '')

# Verify lqns is accessible from Python
import shutil
lqns_exe = shutil.which('lqns')
print(f"lqns found at: {lqns_exe}")

# Use local line_solver from work-dec-8 (with Windows process fix)
sys.path.insert(0, r'C:\Users\gcasale\Dropbox\code\worktrees\work-dec-8\python')
sys.path.insert(0, r'C:\Users\gcasale\Dropbox\experiments\vesta-dev.git')

# Import vesta (which starts JVM internally)
sys.path.insert(0, os.path.abspath("."))  # bench 的上一级
from vesta import *

# Verify which line_solver is being used
import line_solver
print(f"Using line_solver from: {line_solver.__file__}")

# Check Java version being used
import jpype
java_version = str(jpype.java.lang.System.getProperty("java.version"))
print(f"Java version: {java_version}")

# Check what PATH Java sees
java_path = str(jpype.java.lang.System.getenv("PATH"))
print(f"Java sees PATH includes lqns dir: {lqns_path in java_path}")
if lqns_path not in java_path:
    print(f"WARNING: Java PATH does not include {lqns_path}")
    print("You may need to restart your entire computer after adding to system PATH")

import matplotlib.pyplot as plt
import numpy as np
from smt.sampling_methods import LHS, BoxBehnken, PlackettBurman
from smt.surrogate_models import *
from scipy.optimize import *
from scipy.io import savemat, loadmat
import time
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

ndim = 20
ntrain = 3*ndim
ntest = 200
want_train = True

def fun(x):
    t = opt_lqn_20(x)
    return t

def mape(exact, pred):
    mask = exact != 0
    return (np.fabs(exact - pred)/exact)[mask].mean()
def maxape(exact, pred):
    mask = exact != 0
    return (np.fabs(exact - pred)/exact)[mask].max()
xlimits = np.zeros((ndim, 2))
for i in range(int(ndim/2)):
    xlimits[i, 0] = 0.0001
    xlimits[i, 1] = 10.0
    xlimits[10+i, 0] = 1
    xlimits[10+i, 1] = max(2,int(i))

if want_train:
    # Define the function
    sampling = LHS(xlimits=xlimits, criterion='ese')
    xtrain = sampling(ntrain)
    
    # Compute the outputs
    ytrain =  np.zeros(ntrain)
    #print(yt)
    print("Generating training set...")
    for i in range(np.size(xtrain,axis=0)):
        if i % 50 ==0:
            print(i)
        ytrain[i] = fun(xtrain[i,:])
    mdic = {"xtrain": xtrain, "ytrain": ytrain}
    savemat("bench_smt20_train.mat", mdic)
else:
    mat_contents = loadmat("bench_smt20_train.mat")
    xtrain = mat_contents["xtrain"]
    ytrain = np.reshape(mat_contents["ytrain"],(ntrain,1))

print("Generating surrogate...")
labels = ("rbf","krg","kpls","kplsk","mgp","qp","ls","rmtb","idw")
for method in range(8):
    if method == 0: # hangs if ntrain >= 100
        if ntrain < 100:
            sm = RBF(d0=5) # radial basis function
        else:
            continue
    elif method == 1:
        sm = KRG(theta0=[1e-2], corr="pow_exp") # kriging
    elif method == 2:
        sm = KPLS(theta0=[1e-2], corr="pow_exp") # kriging with partial least squares, faster than kriging #corr types: "squar_exp", "pow_exp", "abs_exp"
    elif method == 3:        
        sm = KPLSK(theta0=[1e-2], corr="pow_exp")
    elif method == 4:
        if ntrain < 100: # MGP is very slow otherwise
            sm = MGP(theta0=[1e-2], n_comp=4) # gaussian process
        else:
            continue        
    elif method == 5:
        continue        
        sm = QP() # Second-order polynomial approximation
    elif method == 6:        
        sm = LS() # Least-squares approximation¶
    elif method == 7:        
        sm = RMTB(xlimits=xlimits, order=2, num_ctrl_pts=4, energy_weight=1e-15, regularization_weight=0.01) # Regularized minimal-energy tensor-product splines
    elif method == 8:        
        sm = IDW(p=2) # Inverse distance weighting
    print("method = " + str(labels[method]))
    tic = time.perf_counter()
    sm.options['print_training'] = False
    sm.options['print_prediction'] = False
    sm.options['print_global'] = False
    print(xtrain.shape)
    print(ytrain.shape)
    sm.set_training_values(xtrain, ytrain)
    sm.train()
    toc = time.perf_counter()
    print(f"Time: {toc - tic:0.4f} seconds")

surrfun = lambda x: sm.predict_values(np.array(x.reshape(-1,ndim))).item()
bnds = xlimits
tic = time.perf_counter()

## evolutionary
lc = LinearConstraint([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 10, 10)
res = differential_evolution(surrfun, bnds, maxiter=10000, popsize=50, polish=False, constraints=lc, integrality=(0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1), disp=True)

print(res)
print(-opt_lqn_20(res.x))
toc = time.perf_counter()
print(f"Time: {toc - tic:0.4f} seconds")

print(res.x)

y=res.x
y[19]=10
print(-opt_lqn_20(y))