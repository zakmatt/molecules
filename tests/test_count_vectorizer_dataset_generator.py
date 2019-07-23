import numpy as np
import os
import pandas as pd
import pickle

from numpy.testing import assert_array_equal
from unittest import TestCase
from unittest.mock import patch, MagicMock

from utils.count_vectorizer_dataset_generator import (
    CountVectorizerDatasetGenerator
)


class TestCountVectorizerDatasetGenerator(TestCase):

    def setUp(self):
        current_dir_path = os.path.dirname(__file__)
        self.current_dir_path = current_dir_path
        self.data_path = os.path.join(current_dir_path, 'data/data.csv')
        self.dataset_generator = CountVectorizerDatasetGenerator(
            data_path=self.data_path
        )

    @patch('utils.count_vectorizer_dataset_generator.dump')
    def test_train_vect(self, dump):
        vect = MagicMock()
        vect.fit.return_value = None
        self.dataset_generator._vect = vect
        self.dataset_generator._x_train = pd.DataFrame(
            {
                'smile': [
                    'CC(=O)OC(C)c1ccccc1', 'O=C(O)c1ccccc1S', 'O=CCCc1ccccc1'
                ]
            }
        )

        self.dataset_generator._train_vect()

        self.assertTrue(self.dataset_generator._vect.fit.called)
        self.assertTrue(dump.called)