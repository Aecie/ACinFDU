
#%%
import os
import cupy as cp
import matplotlib.pyplot as plt

from cuml import svm
os.system('export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH')




X = cp.linspace(0, 200, 10000)
Y = cp.sin(X) + cp.random.random(X.shape)

clf = svm.SVC()
clf.fit(X, Y)


print(X.shape, Y.shape)
#%%