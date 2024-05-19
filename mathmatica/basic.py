import cupy as cp
import numpy as np
from typing import Union


def manual_batch_gradient_descendance(X: Union[np.array, cp.array], Y: Union[np.array, cp.array], iterations: int, lr: float, bs: Union[int, tuple]):
    """
    X: sample data
    Y: sample label
    iterations: number of iterations
    lr: learning rate
    bs: batch size
    """


def manual_stochastic_gradient_descendance(X: Union[np.array, cp.array], Y: Union[np.array, cp.array], iterations: int, lr: float, bs: Union[int, tuple]):
    pass