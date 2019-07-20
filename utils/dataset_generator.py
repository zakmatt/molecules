import errno
import logging
import os
import pandas as pd

from abc import ABCMeta, abstractmethod
from rdkit.Chem import rdMolDescriptors, rdmolfiles
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



class DatasetGenerator(metaclass=ABCMeta):
    """"""

    def __init__(self, data_path):
        """

        :param data:
        """

        if not os.path.isfile(data_path):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), data_path
            )

        self.data_path = data_path
        self._read_dataset()

    @property
    def data_path(self):
        """

        :return:
        """

        return self._data_path

    @data_path.setter
    def data_path(self, data_path):
        """

        :param data_path:
        :return:
        """

        if hasattr(self, '_data_path'):
            logging.warning('data_path attribute is already set')
            return

        self._data_path = data_path

    def _read_dataset(self):
        """Dataset reading method"""

        self._dataset = pd.read_csv(self._data_path)

    def _extract_x_y(self):
        """Extract input features and targets"""

        if not hasattr(self, '_dataset'):
            raise AttributeError('dataset not loaded.')

        self._x_dataset = self._dataset['smiles']
        self._y_dataset = self._dataset.drop(columns=['smiles'])

    def divide_into_training_and_test(self, test_size=0.1):
        """Divide dataset into training and test sets

        :param test_size: test set size as a fraction of the whole dataset
        :type test_size: float
        """

        x_train, x_test, y_train, y_test = train_test_split(
            self._x_dataset,
            self._y_dataset,
            test_size=test_size
        )

        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test

    @abstractmethod
    def get_training_data(self):
        pass

    @abstractmethod
    def get_validation_data(self):
        pass

    @abstractmethod
    def get_test_data(self):
        pass
