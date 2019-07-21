import argparse

from models.classification_architectures import ConvNetwork, DenseNetwork
from utils.count_vectorizer_dataset_generator import (
    CountVectorizerDatasetGenerator
)
from utils.morgan_finderprint_dataset_generator import (
    MorganFingerprintDatasetGenerator
)


if __name__ == '__main__':
    dataset_types = [
        CountVectorizerDatasetGenerator, MorganFingerprintDatasetGenerator
    ]
    model_architectures = [
        ConvNetwork, DenseNetwork
    ]

    for dataset_type in dataset_types:
        dataset_generator = dataset_type('./data/data.csv')
        dataset_generator.divide_into_training_and_test()
        for model_architecture in model_architectures:
            dataset_generator.split_into_training_and_validation()
            x_train, y_train = dataset_generator.get_training_data()
            model = model_architecture(
                x_train.shape[1],
                y_train.shape[1],
                dataset_generator.loss_validate_data,
                './results/',
                'train_1.csv'
            )
            model.create_model()
            model.train(x_train, y_train, 1)
            # def __init__(self, input_len, num_of_targets, loss_validate_data,
            # save_model_dir, results_file):
