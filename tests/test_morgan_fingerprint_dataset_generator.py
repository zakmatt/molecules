import numpy as np
import os
import pandas as pd
import pickle

from numpy.testing import assert_array_equal
from rdkit.Chem import rdmolfiles
from unittest import TestCase
from unittest.mock import patch, MagicMock

from utils.morgan_fingerprint_dataset_generator import (
    MorganFingerprintDatasetGenerator
)


class TestMorganFingerprintDatasetGenerator(TestCase):

    def setUp(self):
        current_dir_path = os.path.dirname(os.path.abspath(__file__))
        self.current_dir_path = current_dir_path
        self.data_path = os.path.join(current_dir_path, 'data/data.csv')
        self.dataset_generator = MorganFingerprintDatasetGenerator(
            data_path=self.data_path
        )

    def test_data_path(self):
        self.assertEqual(self.dataset_generator.data_path, self.data_path)

    def test_get_morgan_fingerprint_as_bit_vect(self):
        smile = 'CCCCCCCC(=O)OCC(COC(=O)CCCCCCC)OC(=O)CCCCCCC'
        mol_from_smile = rdmolfiles.MolFromSmiles(smile)
        f_print = self.dataset_generator._get_morgan_fingerprint_as_bit_vect(
            mol_from_smile
        )

        file_path = os.path.join(
            self.current_dir_path, 'data/morgan_fingerprint.pkl'
        )

        with open(file_path, 'rb') as f:
            result_fingerprint = pickle.load(f)

        self.assertEqual(f_print, result_fingerprint)

    def test_morgan_fingerprint_to_binary_vector(self):

        file_path = os.path.join(
            self.current_dir_path, 'data/morgan_fingerprint.pkl'
        )
        with open(file_path, 'rb') as f:
            morgan_fingerprint = pickle.load(f)

        bin_vec = self.dataset_generator._morgan_fingerprint_to_binary_vector(
            morgan_fingerprint
        )

        result_bin_vec = [
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]

        self.assertEqual(bin_vec, result_bin_vec)

    def test_transform_input_features(self):
        smiles = np.array(
            ['CC(=O)OC(C)c1ccccc1', 'O=C(O)c1ccccc1S', 'O=CCCc1ccccc1']
        )
        transformed_smiles = self.dataset_generator._transform_input_features(
            smiles
        )

        expected_transformed = np.array(
            [
                [
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                    0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0
                ],
                [
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ],
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0,
                    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ]
        )

        assert_array_equal(transformed_smiles, expected_transformed)

    def test_extract_x_y(self):
        train_smiles = [
            'CC(=O)OC(C)c1ccccc1', 'O=C(O)c1ccccc1S', 'O=CCCc1ccccc1'
        ]
        train_val_target_1 = [1, 0, 1]
        train_val_target_2 = [0, 0, 1]
        train_val_target_3 = [1, 1, 0]
        train_val_set = pd.DataFrame(
            {
                'smiles' : train_smiles,
                'target_1': train_val_target_1,
                'target_2': train_val_target_2,
                'target_3': train_val_target_3,
            }
        )

        test_set = pd.DataFrame(
            {
                'smiles' : [
                    'CC(=O)O(C)cccc1'
                ],
                'target_1': [1],
                'target_2': [0],
                'target_3': [1],
            }
        )

        self.dataset_generator._train_val_set = train_val_set
        self.dataset_generator._test_set = test_set
        self.dataset_generator.extract_x_y()

        assert_array_equal(
            np.array(self.dataset_generator._x_train_val),
            np.array(train_smiles)
        )
        assert_array_equal(
            self.dataset_generator._y_train_val,
            np.array(
                [train_val_target_1, train_val_target_2, train_val_target_3]
            )
        )
        assert_array_equal(
            np.array(self.dataset_generator._x_test),
            np.array(['CC(=O)O(C)cccc1'])
        )
        assert_array_equal(
            self.dataset_generator._y_test,
            np.array(
                [[1, 0, 1]]
            )
        )

    @patch('utils.dataset_generator.train_test_split')
    @patch('utils.dataset_generator.isfile')
    def test_divide_into_training_and_test(self, isfile, train_test_split):
        isfile.return_value = False # mock isfile to always return False
        train_set = MagicMock('Train Set')
        test_set = MagicMock('Test Set')
        dataset = MagicMock('Dataset')
        self.dataset_generator._dataset = dataset
        train_test_split.return_value = (train_set, test_set)

        self.dataset_generator.divide_into_training_and_test()

        self.assertEqual(
            self.dataset_generator._train_val_set,
            train_set
        )
        self.assertEqual(
            self.dataset_generator._test_set,
            test_set
        )