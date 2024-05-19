"""
Ordinary least squares Linear Regression.
LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.

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
from database.dataset_config import dataset


def example1():
    """
    show the basic parameters after simply train the model
    """
    # set the environment and prepare the data
    use_cuda = True
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



def example2():
    """
    show the plot of fitting result after simply train the model  
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from database.LinearSampleData.LinearSampleData import LinearSampleData

    use_cuda = True
    dataset = LinearSampleData(slope=0.5, bias=-2, samples=20, interval_start=0, interval_end=20, test_rate=0.5, use_cuda=use_cuda)
    train_X, train_Y, test_X, test_Y = dataset.load_data()
    model = CudaLinearRegression() if use_cuda else CPULinearRegression()
    fitted_model = model.fit(train_X.reshape((-1, 1)), train_Y)
    coef, bias = fitted_model.coef_, fitted_model.intercept_

    if use_cuda:
        train_X, train_Y, test_X, test_Y, coef = train_X.get(), train_Y.get(), test_X.get(), test_Y.get(), coef.get()
    plt.figure(figsize=(5, 6))
    plt.scatter(train_X, train_Y, color='#76B900', marker='.', label='train samples')
    plt.scatter(test_X, test_Y, color='#C4C4C4', marker='.', label='test samples')
    line_x = np.linspace(np.concatenate([train_X, test_X]).min(), np.concatenate([train_X, test_X]).max(), np.concatenate([train_X, test_X]).shape[0])
    line_y = coef*line_x + bias
    plt.plot(line_x, line_y, '--', color='#EA4335')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('pred_y=%.2f*x+%.2f' % (coef.item(), bias) if bias > 0 else 'pred_y=%.2f*x' % (coef.item(), bias) if bias == 0 else 'pred_y=%.2f*x%.2f' % (coef.item(), bias))
    plt.show()
