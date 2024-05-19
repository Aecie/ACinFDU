"""
Ridge regression introduces an L2 regulation alpha*||w||^2 in linear least squares

The objective function is ||y - XW||^2_2 + alpha*||W||^2

This model solves a regression model where the loss function is the linear least squares function and regularization is given by the l2-norm. Also known as Ridge Regression or Tikhonov regularization.

[REF] https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
"""


import os
import sys
project_path = os.getcwd()
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from cuml.linear_model import Ridge as CudaRidge
from cuml.pipeline import Pipeline as CudaPipeline
from cuml.preprocessing import PolynomialFeatures as CudaPolynomialFeature
from sklearn.linear_model import Ridge as CPURidge
from sklearn.pipeline import Pipeline as CPUPipeline
from sklearn.preprocessing import PolynomialFeatures as CPUPolynomialFeature


def plot_triple_alphas_effect(train_X, train_Y, use_polynomial: bool, alpha_list: list, style_list: list, use_cuda: bool):
    assert len(alpha_list) == len(style_list)
    for alpha, styple in zip(alpha_list, style_list):
        model = CudaRidge(alpha) if use_cuda else CPURidge(alpha)
        pipeline_params = []
        if use_polynomial:
            pipeline_params.append(('poly_features', CudaPolynomialFeature(degree=5)) if use_cuda else ('poly_features', CPUPolynomialFeature(degree=5)))
        pipeline_params.append(('ridge_regression', model))
        pipeline = CudaPipeline(pipeline_params) if use_cuda else CPUPipeline(pipeline_params)
        pipeline.fit(train_X.reshape((-1, 1)), train_Y)

        linear_space_X = cp.linspace(train_X.min() - .5, train_X.max() + .5, 1000) if use_cuda else cp.linspace(train_X.min() - .5, train_X.max() + .5, 1000)
        linear_space_Y = pipeline.predict(linear_space_X.reshape((-1, 1)))
        if use_cuda:
            plt.plot(linear_space_X.get(), linear_space_Y.get(), styple, label='alpha={}'.format(alpha))
        else:
            plt.plot(linear_space_X, linear_space_Y, styple, label='alpha={}'.format(alpha))
    plt.scatter(train_X.get() if use_cuda else train_X, train_Y.get() if use_cuda else train_Y, color='#76B900', marker='*')
    plt.legend()


def example():
    """
    show the plot of fitting result after simply train the model
    show the effect of alpha value
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from database.LinearSampleData.LinearSampleData import LinearSampleData

    use_cuda = True
    dataset = LinearSampleData(slope=0.5, bias=-2, samples=20, interval_start=0, interval_end=20, test_rate=0.5, use_cuda=use_cuda)
    train_X, train_Y, test_X, test_Y = dataset.load_data()
    print('AAAAAAAAAA', type(train_X), type(train_Y))
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plot_triple_alphas_effect(train_X.copy(), train_Y.copy(), use_polynomial=False, alpha_list=[1e-30, 10, 100], style_list=['b-', 'y--', 'r:'], use_cuda=use_cuda)
    plt.subplot(122)
    plot_triple_alphas_effect(train_X.copy(), train_Y.copy(), use_polynomial=True, alpha_list=[1e-30, .1, 1], style_list=['b-', 'y--', 'r:'], use_cuda=use_cuda)
    plt.savefig('./test_fig.png')
    plt.show()
