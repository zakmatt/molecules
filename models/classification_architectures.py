from keras.layers import (
    Dense, Input, Convolution1D,
    BatchNormalization, AveragePooling1D,
    LeakyReLU
)

from models.classification_model import ClassModel


class DenseNetwork(ClassModel):
    """Dense model class"""

    name = 'DenseNet'

    def _model_architecture(self):
        """Dense model using only dense layers

        :return: computational graph
        :rtype: keras.engine.training.Model
        """

        inputs = Input(shape=self.input_shape)  # (len, )
        x = Dense(1024, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)
        x = Dense(64, activation='relu')(x)

        return inputs, x

    def transform_input_features(self, input_features):
        """Make no transofmration to input features

        :param input_features: input features
        :type input_features: np.array
        :return: input features
        :rtype: input features
        """

        return input_features


class ConvNetwork(ClassModel):
    """Convolution model class"""

    name = 'ConvNet'

    def _model_architecture(self):
        """Convolution model using only convolution layers

        :return: computational graph
        :rtype: keras.engine.training.Model
        """

        inputs = Input(shape=self.input_shape)
        x = Convolution1D(128, 11, strides=1, padding='same')(
            inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = AveragePooling1D(pool_size=5, strides=1, padding='same')(x)
        x = Convolution1D(64, 11, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = AveragePooling1D(pool_size=5, strides=1, padding='same')(x)
        x = Dense(96)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Convolution1D(1, 1)(x)

        return inputs, x

    def transform_input_features(self, input_features):
        """Reshape input features

        :param input_features: model input features
        :type input_features: np.array
        :return: reshaped input features
        :rtype: np.array
        """

        return input_features.reshape(
            input_features.shape[0], 1, input_features.shape[1]
        )
