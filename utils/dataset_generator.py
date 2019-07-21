import errno
import logging
import os
import pandas as pd

from abc import ABCMeta, abstractmethod
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
    """Dataset Generator abstract class


    Dataset Generator abstract class inherited by
    CountVectorizerDatasetGenerator and MorganFingerprintDatasetGenerator
    """

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
        self.divide_into_training_and_test()

        # set attributes with None
        self._x_train_val = None
        self._y_train_val = None
        self._x_test = None
        self._y_test = None
        self._x_train = None
        self._y_train = None
        self._x_val = None
        self._y_val = None
        self._x_test = None
        self._y_test = None

    @property
    def data_path(self):
        """Get data_path attribute

        :return: data_path attribute
        :rtype: str
        """

        return self._data_path

    @data_path.setter
    def data_path(self, data_path):
        """Set data_path attribute

        :param data_path: path to a dataset
        :type data_path: str
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

    def divide_into_training_and_test(self, test_set_size=0.1):
        """Divide dataset into training + validation and test sets

        :param test_set_size: test set size as a fraction of the whole dataset
        :type test_set_size: float
        """

        x_train_val, x_test, y_train_val, y_test = train_test_split(
            self._x_dataset,
            self._y_dataset,
            test_size=test_set_size
        )

        self._x_train_val = x_train_val
        self._y_train_val = y_train_val
        self._x_test = x_test
        self._y_test = y_test

    def split_into_training_and_validation_sets(self, validation_set_size):
        """Divide dataset into training and validation sets

        :param validation_set_size: validation set size as a fraction of
        the whole dataset
        :type validation_set_size: float
        """

        x_train, x_val, y_train, y_val = train_test_split(
            self._x_train_val,
            self._y_train_val,
            test_size=validation_set_size
        )

        self._x_train = x_train
        self._y_train = y_train
        self._x_val = x_val
        self._y_val = y_val

    @abstractmethod
    def _transform_input_features(self, input_features):
        pass

    def get_training_data(self):
        """Get training data; input feature and targets

        :return: input feature and targets
        :rtype: tuple
        """

        return self._transform_input_features(self._x_train), self._y_train

    def get_validation_data(self):
        """Get validation data; input feature and targets

        :return: input feature and targets
        :rtype: tuple
        """

        return self._transform_input_features(self._x_val), self._y_val

    def get_test_data(self):
        """Get test data; input feature and targets

        :return: input feature and targets
        :rtype: tuple
        """

        return self._transform_input_features(self._x_test), self._y_test
