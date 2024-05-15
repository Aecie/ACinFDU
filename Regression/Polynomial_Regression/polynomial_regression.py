"""
Find coefficients w_1, w_2, ..., w_n for n features that minimize the distance (y, y')
    y' = w_1*x^1 + w_2*x^2 + ... + w_d*x^d + bias
target function: t = (1/m)*sigma[1, m](y_i - y'_i)^2

Polynomial Features Parameters
1. degree: int or tuple (min_degree, max_degree), default=2
If a single int is given, it specifies the maximal degree of the polynomial features. If a tuple (min_degree, max_degree) is passed, then min_degree is the minimum and max_degree is the maximum polynomial degree of the generated features. Note that min_degree=0 and min_degree=1 are equivalent as outputting the degree zero term is determined by include_bias.
2. interaction_only: bool, default=False
If True, only interaction features are produced: features that are products of at most degree distinct input features, i.e. terms with power of 2 or higher of the same input feature are excluded:
included: x[0], x[1], x[0] * x[1], etc.
excluded: x[0] ** 2, x[0] ** 2 * x[1], etc.
3. include_bias: bool, default=True
If True (default), then include a bias column, the feature in which all polynomial powers are zero (i.e. a column of ones - acts as an intercept term in a linear model).
4. order{C, F}, default=C
Order of output array in the dense case. 'F' order is faster to compute, but may slow down subsequent estimators.

[REF] https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures
"""


import os
import sys
project_path = os.getcwd()
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from cuml.preprocessing import PolynomialFeatures as CudaPolynomialFeature
from sklearn.preprocessing import PolynomialFeatures as CPUPolynomialFeature
from cuml.linear_model import LinearRegression as CudaLinearRegression
from sklearn.linear_model import LinearRegression as CPULinearRegression

from data.IrisSpecies.Iris_reader import IrisSpeciesReader
from data.BostonHousing.Boston_Housing_reader import BostonHousingReader


# set the environment and prepare the data
use_cuda = True
dataset = BostonHousingReader(use_cuda=use_cuda)
train_X, train_Y, test_X, test_Y = dataset.load_data()

poly_features = CudaPolynomialFeature() if use_cuda else CPUPolynomialFeature()
train_X_poly = poly_features.fit_transform(train_X)
test_X_poly = poly_features.fit_transform(test_X)

model = CudaLinearRegression() if use_cuda else CPULinearRegression()
fitted_model = model.fit(train_X_poly, train_Y)
print('coefficients:', fitted_model.coef_)
print('intercept:', fitted_model.intercept_)

pred_Y = fitted_model.predict(test_X_poly)
MSE = ((pred_Y - test_Y)**2).mean(axis=0)

print('predict result:', pred_Y)
print('test labels:', test_Y)
print('mean squared error:', MSE)