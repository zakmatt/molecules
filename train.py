import argparse

from models.classification_architectures import ConvNetwork, DenseNetwork
from utils.count_vectorizer_dataset_generator import (
    CountVectorizerDatasetGenerator
)
from utils.morgan_finderprint_dataset_generator import (
    MorganFingerprintDatasetGenerator
)


def train_networks(data_path, number_of_epochs):
    dataset_types = [
        CountVectorizerDatasetGenerator, MorganFingerprintDatasetGenerator
    ]
    model_architectures = [
        ConvNetwork,
        DenseNetwork
    ]

    for dataset_type in dataset_types:
        dataset_generator = dataset_type(data_path)
        for model_architecture in model_architectures:
            for i in range(1, 11):
                dataset_generator.divide_into_training_and_validation()
                x_train, y_train = dataset_generator.get_training_data()
                x_val, y_val = dataset_generator.get_training_data()
                model = model_architecture(
                    x_train.shape[1],
                    y_train.shape[1],
                    dataset_generator.loss_validate_data,
                    './results/',
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
                        default='./data/data.csv',
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
