"""
Use polynomial features to extend x with 1 power to n power, where if n is to small, the outcome hyperplane is underfit to samples, otherwise, the hyperplane is overfit, so the selection of n is a trick

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

from cuml.linear_model import LinearRegression as CudaLinearRegression
from cuml.preprocessing import PolynomialFeatures as CudaPolynomialFeature
from sklearn.linear_model import LinearRegression as CPULinearRegression
from sklearn.preprocessing import PolynomialFeatures as CPUPolynomialFeature
from database.dataset_config import dataset


def example1():
    # set the environment and prepare the data
    use_cuda = True
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


def example2():
    """
    show the plot of fitting result after simply train the model  
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from database.SinePlusCosine.SinePlusCosine import SinePlusCosine

    use_cuda = True
    dataset = SinePlusCosine(coef1=1.5, coef2=-0.5, samples=20, interval_start=0, interval_end=2*np.pi, test_rate=0.2, use_cuda=use_cuda)
    train_X, train_Y, test_X, test_Y = dataset.load_data()
    poly_features = CudaPolynomialFeature(degree=3) if use_cuda else CPUPolynomialFeature(degree=3)
    train_X_poly = poly_features.fit_transform(train_X.reshape((-1, 1)))
    model = CudaLinearRegression() if use_cuda else CPULinearRegression()
    fitted_model = model.fit(train_X_poly, train_Y)
    coef, bias = fitted_model.coef_, fitted_model.intercept_

    if use_cuda:
        train_X, train_Y, test_X, test_Y, coef = train_X.get(), train_Y.get(), test_X.get(), test_Y.get(), coef.get()
    plt.figure(figsize=(5, 6))
    print(train_X.shape, train_Y.shape)
    plt.scatter(train_X, train_Y, color='#76B900', marker='.', label='train samples')
    plt.scatter(test_X, test_Y, color='#C4C4C4', marker='.', label='test samples')
    line_x = np.linspace(np.concatenate([train_X, test_X]).min(), np.concatenate([train_X, test_X]).max(), np.concatenate([train_X, test_X]).shape[0])
    line_y = coef[-1]*np.power(line_x, 3) + coef[-2]*np.power(line_x, 2) + coef[-3]*line_x + bias
    plt.plot(line_x, line_y, '--', color='#EA4335')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('pred_y=(%.2f)*x^3+(%.2f)*x^2+(%.2f)*x+(%.2f)' % (coef[-1], coef[-2], coef[-3], bias))
    # plt.savefig('./test_fig.png')
    plt.show()
