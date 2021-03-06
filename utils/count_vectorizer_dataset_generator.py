import errno
import os

from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer

from utils.dataset_generator import DatasetGenerator


class CountVectorizerDatasetGenerator(DatasetGenerator):
    """Dataset generator class using CountVectorizer
    in order to transform input features
    """

    name = 'count_vectorizer_data_generator'

    def __init__(self, data_path):
        """Initialization method

        :param data_path: path to a csv file containing data
        :type data_path: str
        """

        DatasetGenerator.__init__(self, data_path)

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
        dump(self._vect, save_path, compress=9)

    def load_vect(self, path):
        """Load pickled, pre-trained count vectorizer

        :param path: pre-trained, pickled count vectorizer file path
        :type path: str
        """

        if not os.path.isfile(path):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), path
            )

        self._vect = load(path)

    def divide_into_training_and_validation(self, validation_set_size=0.1):
        """Divide dataset into training and validation sets

        :param validation_set_size: validation set size as a fraction of
        the whole dataset
        :type validation_set_size: float
        """

        super().divide_into_training_and_validation(validation_set_size)
        self._train_vect()

    def _transform_input_features(self, input_features):
        """Transfer input features using count vectorizer

        :param input_features: input features
        :type input_features: np.array
        :return: transformed input features
        :rtype: np.array
        """

        return self._vect.transform(input_features).toarray()
