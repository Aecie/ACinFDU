"""
Find coefficients w_1, w_2, ..., w_n for n features that minimize the distance (y, y')
    y' = x_1*w_1 + x_2*w_2 + ... + x_n*w_n + bias
target function: t = (1/m)*sigma[1, m](y_i - y'_i)^2

Linear Regression Parameters
1. fit_intercept: bool, default=True
Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
2. copy_X: bool, default=True
If True, X will be copied; else, it may be overwritten.
3. n_jobs: int, default=None
The number of jobs to use for the computation. This will only provide speedup in case of sufficiently large problems, that is if firstly n_targets > 1 and secondly X is sparse or if positive is set to True. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.
4. positive: bool, default=False
When set to True, forces the coefficients to be positive. This option is only supported for dense arrays.

[REF] https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
"""


import os
import sys
project_path = os.getcwd()
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from cuml.linear_model import LinearRegression as CudaLinearRegression
from sklearn.linear_model import LinearRegression as CPULinearRegression

from data.IrisSpecies.Iris_reader import IrisSpeciesReader
from data.BostonHousing.Boston_Housing_reader import BostonHousingReader


# set the environment and prepare the data
use_cuda = True
dataset = BostonHousingReader(use_cuda=use_cuda)
train_X, train_Y, test_X, test_Y = dataset.load_data()

model = CudaLinearRegression() if use_cuda else CPULinearRegression()
fitted_model = model.fit(train_X, train_Y)
print('coefficients:', fitted_model.coef_)
print('intercept:', fitted_model.intercept_)

pred_Y = fitted_model.predict(test_X)
MSE = ((pred_Y - test_Y)**2).mean(axis=0)

print('predict result:', pred_Y)
print('test labels:', test_Y)
print('mean squared error:', MSE)