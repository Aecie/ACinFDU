import random
import cupy as cp
import numpy as np


class SinePlusCosine():
    def __init__(self, coef1: float=1., coef2: float=1., samples: int=10000, interval_start: float=-100, interval_end: float=100, test_rate: float=.1, use_cuda: bool=False) -> None:
        random.seed('this is a random number')
        self.__use_cuda = use_cuda
        self.__test_rate = test_rate
        self.__train_X = np.linspace(interval_start, interval_end, samples)
        self.__train_Y = coef1*np.cos(self.__train_X) + coef2*np.sin(self.__train_X) + np.random.normal(0, 1, self.__train_X.shape)
        self.__test_X = np.linspace(interval_end, interval_end + (interval_end - interval_start)*self.__test_rate/(1 - self.__test_rate), int(samples*self.__test_rate))
        self.__test_Y = coef1*np.cos(self.__test_X) + coef2*np.sin(self.__test_X) + np.random.normal(0, 1, self.__test_X.shape)
    
    def load_data(self):
        if self.__use_cuda:
            return cp.array(self.__train_X), cp.array(self.__train_Y), cp.array(self.__test_X), cp.array(self.__test_Y)
        else:
            return np.array(self.__train_X), np.array(self.__train_Y), np.array(self.__test_X), np.array(self.__test_Y)