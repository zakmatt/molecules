import argparse
import os

from models.classification_architectures import ConvNetwork, DenseNetwork
from utils.count_vectorizer_dataset_generator import (
    CountVectorizerDatasetGenerator
)
from utils.morgan_fingerprint_dataset_generator import (
    MorganFingerprintDatasetGenerator
)

CURRENT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))


def train_networks(data_path, number_of_epochs):
    """Training method

    Training method which trains, saves and evaluates models.

    :param data_path: Path to a dataset
    :type data_path: str
    :param number_of_epochs: number of training epochs
    :type number_of_epochs: int
    """

    dataset_types = [
        CountVectorizerDatasetGenerator, MorganFingerprintDatasetGenerator
    ]
    model_architectures = [
        ConvNetwork,
        DenseNetwork
    ]

    save_dataset = True
    for dataset_type in dataset_types:
        dataset_generator = dataset_type(data_path)
        dataset_generator.read_dataset()
        dataset_generator.divide_into_training_and_test()
        if save_dataset:
            save_dataset = False
            dataset_generator.save_train_test_sets()
        dataset_generator.extract_x_y()
        for model_architecture in model_architectures:
            for i in range(1, 6):
                dataset_generator.divide_into_training_and_validation()
                x_train, y_train = dataset_generator.get_training_data()
                x_val, y_val = dataset_generator.get_training_data()
                model = model_architecture(
                    x_train.shape[1],
                    y_train.shape[1],
                    dataset_generator.loss_validate_data,
                    './results/{}'.format(dataset_generator.name),
                    'train_{}.csv'.format(i)
                )
                model.create_model()
                model.train(x_train, y_train, x_val, y_val, number_of_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--data_path",
                        help='Training dataset csv file. '
                             'Default: ./data/data.csv',
                        default=os.path.join(
                            CURRENT_DIR_PATH, 'data/data.csv'
                        ),
                        required=False)
    parser.add_argument("-e",
                        "--epochs",
                        help='Number of epochs. Default: 50.',
                        default=50,
                        required=False)

    args = parser.parse_args()
    data_path = args.data_path
    number_of_epochs = int(args.epochs)

    train_networks(data_path, number_of_epochs)
