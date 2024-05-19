import os
import sys

from database.BostonHousing.Boston_Housing_reader import BostonHousingReader
from database.FashionMNIST.FashionMNIST_reader import FashionMNIST_Reader
from database.IrisSpecies.Iris_reader import IrisSpeciesReader
from database.MNIST.MNIST_reader import MNIST_Reader
from database.SinePlusCosine.SinePlusCosine import SinePlusCosine
from database.LinearSampleData.LinearSampleData import LinearSampleData

project_path = os.getcwd()
if project_path not in sys.path:
    sys.path.insert(0, project_path)
dataset = BostonHousingReader()


