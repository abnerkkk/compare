import os, sys
sys.path.insert(0, os.path.abspath("."))
from vesta import *
import matplotlib.pyplot as plt
import numpy as np
from smt.sampling_methods import LHS
from smt.surrogate_models import *
from scipy.optimize import *
from scipy.io import savemat, loadmat
import time
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

ndim = 2
ntrain = 500
ntest = 200
want_train = True
want_true = True

def fun(x):
    t = opt_lqn_2(x)
    return t

def mape(exact, pred):
    mask = exact != 0
    return (np.fabs(exact - pred)/exact)[mask].mean()
def maxape(exact, pred):
    mask = exact != 0
    return (np.fabs(exact - pred)/exact)[mask].max()

xlimits = np.zeros((ndim, 2))
xlimits[:, 0] = 0.0
xlimits[:, 1] = 100.0
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
    savemat("bench_smt_train.mat", mdic)
else:
    mat_contents = loadmat("bench_smt_train.mat")
    xtrain = mat_contents["xtrain"]
    ytrain = np.reshape(mat_contents["ytrain"],(ntrain,1))

    #print("Generating test set...")
xtestlimits = xlimits
sampling = LHS(xlimits=xtestlimits, criterion='ese', random_state=1)
xtest = sampling(ntest)
#print("Generating predictions...")
ypred = np.zeros(ntest)
ytrue = np.zeros(ntest)
if want_true:
    print("Generating test data...")
    for i in range(np.size(xtest,axis=0)):
        if i % 50 ==0:
            print(i)
        ytrue[i] = fun(xtest[i,:])
        mdic = {"xtest": xtest, "ytrue": ytrue}
        savemat("bench_smt_test.mat", mdic)
else:
    mat_contents = loadmat("bench_smt_test.mat")
    xtest = mat_contents["xtest"]
    ytrue = np.reshape(mat_contents["ytrue"],(ntest,1))

    print("Generating surrogate...")
labels = ("rbf","krg","kpls","kplsk","mgp","qp","ls","rmtb","idw")
for method in range(2):
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
        sm = QP() # Second-order polynomial approximation
    elif method == 6:        
        sm = LS() # Least-squares approximation¶
    elif method == 7:        
        sm = RMTB(xlimits=xlimits, order=2, num_ctrl_pts=20, energy_weight=1e-15, regularization_weight=0.0) # Regularized minimal-energy tensor-product splines
    elif method == 8:        
        sm = IDW(p=2) # Inverse distance weighting
    print("method = " + str(labels[method]))
    bnds = ((0.001, 10), (0.001, 15))
    tic = time.perf_counter()
    sm.options['print_training'] = False
    sm.options['print_prediction'] = False
    sm.options['print_global'] = False
    sm.set_training_values(xtrain, ytrain)
    sm.train()
    toc = time.perf_counter()
    print(f"Time: {toc - tic:0.4f} seconds")
    for i in range(np.size(xtest,axis=0)):
        ypred[i] = sm.predict_values(np.array(xtest[i,:].reshape(-1,2))).item()
    ypred = np.reshape(ypred,(ntest,1))
    print("MAPE=" + str(mape(exact=ytrue,pred=ypred)))
    print("MaxAPE=" + str(maxape(exact=ytrue,pred=ypred)))
    if method == 2:
        plt.plot(ypred, "o")
        plt.plot(ytrue)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(["Prediction", "Exact"])
        plt.title(str(labels[method]))
        plt.show()

surrfun = lambda x: -sm.predict_values(np.array(x.reshape(-1,2))).item()
cons = ({'type': 'ineq', 'fun': lambda x: 10 - x[0] - x[1]}, {'type': 'ineq', 'fun': lambda x: -(10 - x[0] - x[1])})
bnds = ((0.001, 10), (0.001, 15))
tic = time.perf_counter()
res = minimize(surrfun, (0.001, 0.001), method='COBYLA', bounds=bnds, constraints=cons, tol=1e-6, options={"maxiter": 500, "disp": True})
#res = minimize(lambda x: -fun(x), (0.001, 0.001), method='COBYLA', bounds=bnds, constraints=cons, tol=1e-6, options={"maxiter": 500, "disp": True})
toc = time.perf_counter()
print(f"Time: {toc - tic:0.4f} seconds")
xopt=res.x
yopt=fun(res.x)
print(xopt)
print(yopt)