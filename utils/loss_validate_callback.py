import numpy as np
import os
from keras.callbacks import Callback


class LossValidateCallback(Callback):
    """"""

    def __init__(self, loss_validate_data, transform_features, results_file):
        """

        :param loss_validate_data:
        :param results_file:
        """

        x_train, y_train, x_val, y_val = loss_validate_data()
        self.transform_input_features = transform_features
        n = x_val.shape[0]  # number of rows
        index = np.random.choice(x_train.shape[0], n, replace=False)
        self.x_train = x_train[index]
        self.y_train = y_train[index]
        self.x_val = x_val
        self.y_val = y_val
        self.val_loss = 0

        basedir = os.path.dirname(results_file)
        if not os.path.exists(basedir):
            os.makedirs(basedir)
        self.results_file = results_file

    def on_epoch_end(self, epoch, logs=None):
        train_loss, train_acc = self.model.evaluate(
            self.transform_input_features(self.x_train), self.y_train
        )
        val_loss, val_acc = self.model.evaluate(
            self.transform_input_features(self.x_val), self.y_val
        )
        self.val_loss = val_loss

        text = '{0}, {1}, {2}, {3}, {4}\n'.format(
            epoch, train_loss, train_acc, val_loss, val_acc
        )

        if not os.path.exists(self.results_file):
            with open(self.results_file, 'w') as file:
                columns = 'epoch, train_loss, train_accuracy, ' \
                          'validation_loss, validation_accuracy\n'
                file.writelines(columns)

        with open(self.results_file, 'a') as file:
            file.writelines(text)
