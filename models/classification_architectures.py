from keras.layers import (
    Dense, Input, Convolution1D,
    BatchNormalization, AveragePooling1D,
    LeakyReLU
)

from models.classification_model import ClassModel


class DenseNetwork(ClassModel):
    """ """

    name = 'DenseNet'

    def _model_architecture(self):
        """

        :return:
        """

        inputs = Input(shape=self.input_shape) # (len, )
        x = Dense(1024, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)
        x = Dense(64, activation='relu')(x)

        return inputs, x


class ConvNetwork(ClassModel):
    """ """

    name = 'ConvNet'

    def _model_architecture(self):
        """

        :return:
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