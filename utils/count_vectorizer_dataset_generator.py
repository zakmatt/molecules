import errno
import joblib
import numpy as np
import os

from sklearn.feature_extraction.text import CountVectorizer

from utils.dataset_generator import DatasetGenerator


class CountVectorizerDatasetGenerator(DatasetGenerator):
    """Dataset generator class using CountVectorizer
    in order to transform input features
    """

    def __init__(self, data_path):
        """

        :param data_path:
        """

        DatasetGenerator.__init__(data_path)

        # count vectorizer object used to transform input data
        self._vect = CountVectorizer(
            analyzer=u'char',
            lowercase=False,
            ngram_range=(1, 1)
        )

    def _train_vect(self):
        """Train count vectorizer on x_train"""

        self._vect.fit(self._x_train)

        current_dir_path = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir_path, 'count_vectorizer.pkl')
        joblib.dump(self._vect, save_path, compress=9)

    def loac_vect(self, path):
        """Load pickled, pre-trained count vectorizer

        :param path: pre-trained, pickled count vectorizer file path
        :type path: str
        """

        if not os.path.isfile(path):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), path
            )

        self._vect = joblib.load(path)

    def split_into_training_and_validation_sets(self, validation_set_size):
        """Divide dataset into training and validation sets

        :param validation_set_size: validation set size as a fraction of
        the whole dataset
        :type validation_set_size: float
        """

        super().split_into_training_and_validation_sets(validation_set_size)
        self._train_vect()

    def _transform_input_features(self, input_features):
        """Transfer input features using count vectorizer

        :param input_features: input features
        :type input_features: np.array
        :return: transformed input features
        :rtype: np.array
        """

        return self._vect.transform(input_features)
