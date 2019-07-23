import numpy as np

from rdkit.Chem import rdMolDescriptors, rdmolfiles

from utils.dataset_generator import DatasetGenerator

NUMBER_OF_BITS = 256


class MorganFingerprintDatasetGenerator(DatasetGenerator):
    """Dataset generator class using binary representation of Morgan
    Fingerprint in order to transform input features
    """

    name = 'morgan_fingerprint_data_generator'

    @staticmethod
    def _morgan_fingerprint_to_binary_vector(morgan_fingerprint):
        """Convert morgan fingerprint in a

        :param morgan_fingerprint: Morgan Fingerprint
        :type morgan_fingerprint: rdkit.DataStructs
        .cDataStructs.ExplicitBitVect
        :return: binary vector representation of a Morgan Fingerprint object
        :rtype: list
        """

        bit_string = morgan_fingerprint.ToBitString()
        binary_vector = list(map(int, bit_string))

        return binary_vector

    @staticmethod
    def _get_morgan_fingerprint_as_bit_vect(mol_smile):
        """Convert mol representation to Morgan Fingerprint

        :param mol_smile: mol representation of smile
        :type mol_smile: rdkit.Chem.rdchem.Mol
        :return: Morgan Fingerprint
        :rtype: rdkit.DataStructs.cDataStructs.ExplicitBitVect
        """

        bit_info = {}
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol_smile, radius=2, bitInfo=bit_info, nBits=NUMBER_OF_BITS
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
            np.array(
                self._morgan_fingerprint_to_binary_vector(morgan_fingerprint)
            ) for morgan_fingerprint in morgan_fingerprints
        ]

        return np.array(binary_vecotrs)
