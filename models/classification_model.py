import keras.backend as K
import logging
import os

from abc import ABCMeta, abstractmethod
from keras.callbacks import ReduceLROnPlateau
from keras.layers import (
    Dense, Reshape
)
from keras.models import Model

from utils.loss_validate_callback import LossValidateCallback

MASK_VALUE = -1

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ADD SOURCE!!!
def build_masked_loss(loss_function, mask_value=MASK_VALUE):
    """Builds a loss function that masks based on targets

    Args:
        loss_function: The loss function to mask
        mask_value: The value to mask in the targets

    Returns:
        function: a loss function that acts like loss_function with masked inputs
    """

    def masked_loss_function(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        return loss_function(y_true * mask, y_pred * mask)

    return masked_loss_function


def masked_accuracy(y_true, y_pred):
    dtype = K.floatx()
    total = K.sum(K.cast(K.not_equal(y_true, MASK_VALUE), dtype))
    correct = K.sum(K.cast(K.equal(y_true, K.round(y_pred)), dtype))
    return correct / total


class MissingTestSetException(Exception):
    pass


# noinspection PyPep8Naming
class ClassModel(metaclass=ABCMeta):

    name = 'class_model'

    def __init__(self, input_shape, num_of_targets, loss_validate_data,
                 save_model_dir, results_file):
        """

        :param input_shape:
        :type input_shape:
        :param num_of_targets:
        :type num_of_targets:
        :param loss_validate_data:
        :type loss_validate_data:
        :param save_model_dir:
        :type save_model_dir:
        :param results_file:
        :type results_file:
        """

        self.input_shape = input_shape
        self.num_of_targets = num_of_targets
        self.model = None

        reduce_on_plateau = ReduceLROnPlateau(
            monitor='val_loss', factor=0.8, patience=10, verbose=1,
            mode='auto', min_delta=0.0001, cooldown=5, min_lr=0.0001
        )

        results_file = os.path.join(save_model_dir, results_file)

        self.callbacks = [
            reduce_on_plateau,
            LossValidateCallback(
                loss_validate_data,
                results_file
            )
        ]

    @abstractmethod
    def _model_architecture(self):
        pass

    def create_model(self):
        """Create a fine tuned model based on the provided architecture
        :param img_rows: number of rows in an image
        :type img_rows: int
        :param img_cols: number of columns in an image
        :type img_cols: int
        :param model_architecture: model architecture
        :type model_architecture: keras.applications.vgg16.VGG16
        """

        inputs, network_outputs = self._model_architecture()

        network_outputs = Dense(
            self.num_of_targets, activation='sigmoid'
        )(network_outputs)
        network_outputs = Reshape(target_shape=(self.num_of_targets,))(
            network_outputs)
        model = Model(inputs=inputs, outputs=network_outputs)
        model.compile(
            loss=build_masked_loss(K.binary_crossentropy),
            optimizer='adam', metrics=[masked_accuracy]
        )

        self.model = model

    def train(self, x_train, y_train, nb_epochs):
        """Training method

        :param x_train: input features
        :type x_train: np.array
        :param y_train: output targets
        :type y_train: np.array
        :param nb_epochs: number of epochs
        :type nb_epochs: int
        """

        x_train = self.transform_input_features(x_train)
        if hasattr(self, 'model'):
            self.model.fit(
                x=x_train,
                y=y_train,
                epochs=nb_epochs,
                callbacks=self.callbacks,
                batch_size=32
            )

    def load_weights(self, weights_path):
        if hasattr(self, 'model'):
            self.model.load_weights(weights_path)

    @abstractmethod
    def transform_input_features(self, input_features):
        pass

    def predict(self, input_features):
        """

        :param input_features:
        :return:
        """

        transformed_features = self.transform_input_features(input_features)

        self.model.predict(transformed_features)

    def evaluate_on_test_set(self, x_test, y_test):

        x_test = self.transform_input_features(x_test)
        loss, accuracy = self.model.evaluate(x_test, y_test)

        return loss, accuracy
