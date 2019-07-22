import argparse
import os

from models.classification_architectures import ConvNetwork, DenseNetwork
from utils.count_vectorizer_dataset_generator import (
    CountVectorizerDatasetGenerator
)
from utils.morgan_fingerprint_dataset_generator import (
    MorganFingerprintDatasetGenerator
)


class WrongModelTypeException(Exception):
    pass


class WrongInputFeaturesTypeException(Exception):
    pass


def evaluate(model_type, features_type, test_data_path):
    if features_type == 'morgan':
        data_gen = MorganFingerprintDatasetGenerator(test_data_path)
        weights_features_type = 'morgan_fingerprint_data_generator'
    else:
        data_gen = CountVectorizerDatasetGenerator(test_data_path)
        current_dir_path = os.path.dirname(os.path.abspath(__file__))
        vect_path = os.path.join(
            current_dir_path, 'utils/count_vectorizer.pkl'
        )
        data_gen.load_vect(vect_path)
        weights_features_type = 'count_vectorizer_data_generator'

    data_gen.read_test_dataset()

    x_test, y_test = data_gen.get_test_data()

    if model_type == 'ConvNet':
        model = ConvNetwork
        weights_model = 'convnet'
    else:
        model = DenseNetwork
        weights_model = 'densenet'

    weights_path = './results/{}/{}/best_model.hdf5'.format(
        weights_features_type, weights_model
    )
    model = model(
        x_test.shape[1],
        y_test.shape[1],
        evaluate=True
    )
    model.create_model()
    model.load_weights(weights_path)

    loss, accuracy = model.evaluate_on_test_set(x_test, y_test)

    print(
        '{} model loss and accuracy equal {:.2f} and {:.2f}, respectively'.format(
            model_type, loss, accuracy
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m",
                        "--model_type",
                        help='Model type. Either ConvNet or DenseNet',
                        default='DenseNet',
                        required=False)
    parser.add_argument("-f",
                        "--features_type",
                        help='Input features type. Either morgan (using '
                             'binary representation of Morgan Fingerprint) or '
                             'vect (Count Vectorizer)',
                        default='morgan',
                        required=False)
    parser.add_argument("-t",
                        "--test_data_path",
                        help='test file data path',
                        default='./data/test_set.csv',
                        required=False)

    args = parser.parse_args()
    model_type = args.model_type
    if model_type not in ['ConvNet', 'DenseNet']:
        raise WrongModelTypeException()
    features_type = args.features_type
    if features_type not in ['morgan', 'vect']:
        raise WrongInputFeaturesTypeException()
    test_data_path = args.test_data_path
    if not os.path.isfile(test_data_path):
        raise FileNotFoundError()

    evaluate(model_type, features_type, test_data_path)
