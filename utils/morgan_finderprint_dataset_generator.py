import numpy as np

from rdkit.Chem import rdMolDescriptors, rdmolfiles

from utils.dataset_generator import DatasetGenerator


class MorganFingerprintDatasetGenerator(DatasetGenerator):
    """Dataset generator class using binary representation of Morgan
    Fingerprint in order to transform input features
    """

    @staticmethod
    def _morgan_fingerprint_to_binary_vector(morgan_fingerprint):
        """

        :param morgan_fingerprint:
        :return:
        """

        bit_string = morgan_fingerprint.ToBitString()
        binary_vector = list(map(int, bit_string))

        return binary_vector

    @staticmethod
    def _get_morgan_fingerprint_as_bit_vect(mol_smile):
        """Get Morgan fingerprint as bit vector

        :param mol_smile: mol representation of smile
        :return: list
        """

        bit_info = {}
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol_smile, radius=2, bitInfo=bit_info, nBits=256
        )

    def _transform_input_features(self, input_features):
        """Transform input features into morgan fingerprint binary vectors

        :param input_features: input features
        :type input_features: np.array
        :return: binary vectors of 256 elements
        :rtype: np.array
        """

        mol_smiles = [
            rdmolfiles.MolFromSmiles(
                input_feature
            ) for input_feature in input_features
        ]

        morgan_fingerprints = [
            self._get_morgan_fingerprint_as_bit_vect(
                mol_smile
            ) for mol_smile in mol_smiles
        ]

        binary_vecotrs = [
            self._morgan_fingerprint_to_binary_vector(
                morgan_fingerprint
            ) for morgan_fingerprint in morgan_fingerprints
        ]

        return np.array(binary_vecotrs)
