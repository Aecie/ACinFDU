"""
This is an Iris Species data reader to get train data and test data
"""


import os
import random
import cupy as cp
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

class IrisSpeciesReader():
    def __init__(self, test_rate: float=.1, use_cuda: bool=False) -> None:
        """
        test_rate: the proportion of test data
        """
        random.seed('this is a random number')
        self.__use_cuda = use_cuda
        self.__test_rate = test_rate
        self.__file_path = os.path.join(os.getcwd(), os.path.join('data/IrisSpecies', 'Iris.csv'))
        # self.load_data()
        # self.__train_X, self.__train_Y, self.__test_X, self.__test_Y = self.load_data()
        # print(self.__train_X.shape, self.__train_Y.shape, self.__test_X.shape, self.__test_Y.shape)

    def load_data(self):        
        df = pd.read_csv(self.__file_path)
        N = df.shape[0]
        test_N = int(N * self.__test_rate)
        test_indices = np.array(random.sample([i for i in range(N)], k=test_N))
        train_indices = np.array([idx for idx in range(N) if idx not in test_indices])

        features = df.iloc[:, : -1].to_numpy()  # the non-last columns are features
        species = df.iloc[:, -1:].to_numpy()  # the last column is the species
        labels = LabelEncoder().fit_transform(species.ravel())
        train_X, train_Y, test_X, test_Y = features[train_indices], labels[train_indices], features[test_indices], labels[test_indices]
        
        if self.__use_cuda:
            return cp.array(train_X), cp.array(train_Y), cp.array(test_X), cp.array(test_Y)
        else:
            return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)