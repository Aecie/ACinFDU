"""
This is a MNIST data reader to get train data and test data
"""


import os
import struct
import cupy as cp
import numpy as np
from array import array


class MNIST_Reader():
    def __init__(self, train_scale: float=1.0, test_scale: float=1.0, use_cuda: bool=False) -> None:
        """
        train_scale: get the scale of the train dataset, range from 0.0 to 1.0
        test_scale: get the scale of the test dataset, range from 0.0 to 1.0
        """
        self.__use_cuda = use_cuda
        self.__train_scale, self.__test_scale = train_scale, test_scale
        self.__train_image_path = os.path.join(os.getcwd(), os.path.join('database/MNIST', 'train-images.idx3-ubyte'))
        self.__train_label_path = os.path.join(os.getcwd(), os.path.join('database/MNIST', 'train-labels.idx1-ubyte'))
        self.__test_image_path = os.path.join(os.getcwd(), os.path.join('database/MNIST', 't10k-images.idx3-ubyte'))
        self.__test_label_path = os.path.join(os.getcwd(), os.path.join('database/MNIST', 't10k-labels.idx1-ubyte'))
        self.__train_X, self.__train_Y, self.__test_X, self.__test_Y = self.load_data()
        print(self.__train_X.shape, self.__train_Y.shape, self.__test_X.shape, self.__test_Y.shape)

    def read_images_labels(self, images_file_path, labels_file_path):        
        labels = []
        with open(labels_file_path, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_file_path, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        if self.__use_cuda:
            return cp.array(images), cp.array(labels)
        else:
            return np.array(images), np.array(labels)

    def set_cuda(self, use_cuda: bool) -> None:
        self.__use_cuda = use_cuda

    def load_data(self) -> tuple:
        train_X, train_Y = self.read_images_labels(self.__train_image_path, self.__train_label_path)
        test_X, test_Y = self.read_images_labels(self.__test_image_path, self.__test_label_path)
        train_N, test_N = int(train_X.shape[0]*self.__train_scale), int(test_X.shape[0]*self.__test_scale)
        return train_X[:train_N], train_Y[:train_N], test_X[:test_N], test_Y[:test_N]