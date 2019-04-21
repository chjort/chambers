import tensorflow as tf
from tensorflow.python.keras import layers


class UpsampleNN(layers.Layer):
    def __init__(self, scale):
        super(UpsampleNN, self).__init__()
        self.scale = scale

    def build(self, input_shape):
        self.w = input_shape[1] * self.scale
        self.h = input_shape[2] * self.scale

    def call(self, input, **kwargs):
        return tf.image.resize_nearest_neighbor(input, size=[self.w, self.h])

    def compute_output_shape(self, input_shape):
        return (self.w, self.h)


class UnpoolBilinear(layers.Layer):
    def __init__(self, scale):
        super(UnpoolBilinear, self).__init__()
        self.scale = scale

    def build(self, input_shape):
        self.w = input_shape[1] * self.scale
        self.h = input_shape[2] * self.scale

    def call(self, input, **kwargs):
        return tf.image.resize_bilinear(input, size=[self.w, self.h])

    def compute_output_shape(self, input_shape):
        return (self.w, self.h)


def ResidualUnit(x, filters=48, filter_size=3):
    x = layers.Conv2D(filters, kernel_size=[filter_size, filter_size], padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size=[filter_size, filter_size], padding="same")(x)
    x = layers.BatchNormalization()(x)

    return x


def FullResolutionResidualUnit(y, z, filters, pool_scale):
    FILTERS_1X1 = 32

    G = layers.MaxPool2D(pool_size=[pool_scale, pool_scale], strides=[pool_scale, pool_scale], padding="same")(z)
    G = layers.Concatenate()([y, G])

    G = layers.Conv2D(filters, kernel_size=[3, 3], padding="same")(G)
    G = layers.BatchNormalization()(G)
    G = layers.ReLU()(G)

    G = layers.Conv2D(filters, kernel_size=[3, 3], padding="same")(G)
    G = layers.BatchNormalization()(G)
    G = layers.ReLU()(G)

    H = layers.Conv2D(FILTERS_1X1, kernel_size=[1, 1], padding="same")(G)
    H = UpsampleNN(pool_scale)(H)
    H = layers.Add()([z, H])

    return G, H


def FRRN_A(inputs, num_classes):

    with tf.name_scope("FRRN_A"):
        # Initial Stage
        x = layers.Conv2D(48, kernel_size=[5, 5], padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = ResidualUnit(x, filters=48, filter_size=3)
        x = ResidualUnit(x, filters=48, filter_size=3)
        x = ResidualUnit(x, filters=48, filter_size=3)

        # Downsampling Path
        y = layers.MaxPool2D(pool_size=[2, 2], padding="same")(x)
        z = layers.Conv2D(32, kernel_size=[1, 1], padding="same")(x)

        y, z = FullResolutionResidualUnit(y, z, 96, 2)
        y, z = FullResolutionResidualUnit(y, z, 96, 2)
        y, z = FullResolutionResidualUnit(y, z, 96, 2)
        y = layers.MaxPool2D(pool_size=[2, 2], padding="same")(y)

        y, z = FullResolutionResidualUnit(y, z, 192, 4)
        y, z = FullResolutionResidualUnit(y, z, 192, 4)
        y, z = FullResolutionResidualUnit(y, z, 192, 4)
        y, z = FullResolutionResidualUnit(y, z, 192, 4)
        y = layers.MaxPool2D(pool_size=[2, 2], padding="same")(y)

        y, z = FullResolutionResidualUnit(y, z, 384, 8)
        y, z = FullResolutionResidualUnit(y, z, 384, 8)
        y = layers.MaxPool2D(pool_size=[2, 2], padding="same")(y)

        y, z = FullResolutionResidualUnit(y, z, 384, 16)
        y, z = FullResolutionResidualUnit(y, z, 384, 16)

        # Upsampling Path
        y = UnpoolBilinear(2)(y)
        y, z = FullResolutionResidualUnit(y, z, 192, 8)
        y, z = FullResolutionResidualUnit(y, z, 192, 8)

        y = UnpoolBilinear(2)(y)
        y, z = FullResolutionResidualUnit(y, z, 192, 4)
        y, z = FullResolutionResidualUnit(y, z, 192, 4)

        y = UnpoolBilinear(2)(y)
        y, z = FullResolutionResidualUnit(y, z, 96, 2)
        y, z = FullResolutionResidualUnit(y, z, 96, 2)

        y = UnpoolBilinear(2)(y)

        # Final Stage
        x = layers.Concatenate()([y, z])
        x = ResidualUnit(x, filters=48, filter_size=3)
        x = ResidualUnit(x, filters=48, filter_size=3)
        x = ResidualUnit(x, filters=48, filter_size=3)

        x = layers.Conv2D(num_classes, kernel_size=[1, 1], padding="same")(x)
        x = layers.Softmax()(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x, name="FRRN-A")

    return model

def FRRN_B(inputs, num_classes):

    with tf.name_scope("FRRN_B"):
        # Initial Stage
        x = layers.Conv2D(48, kernel_size=[5, 5], padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = ResidualUnit(x, filters=48, filter_size=3)
        x = ResidualUnit(x, filters=48, filter_size=3)
        x = ResidualUnit(x, filters=48, filter_size=3)

        # Downsampling Path
        y = layers.MaxPool2D(pool_size=[2, 2], padding="same")(x)
        z = layers.Conv2D(32, kernel_size=[1, 1], padding="same")(x)

        y, z = FullResolutionResidualUnit(y, z, 96, 2)
        y, z = FullResolutionResidualUnit(y, z, 96, 2)
        y, z = FullResolutionResidualUnit(y, z, 96, 2)
        y = layers.MaxPool2D(pool_size=[2, 2], padding="same")(y)

        y, z = FullResolutionResidualUnit(y, z, 192, 4)
        y, z = FullResolutionResidualUnit(y, z, 192, 4)
        y, z = FullResolutionResidualUnit(y, z, 192, 4)
        y, z = FullResolutionResidualUnit(y, z, 192, 4)
        y = layers.MaxPool2D(pool_size=[2, 2], padding="same")(y)

        y, z = FullResolutionResidualUnit(y, z, 384, 8)
        y, z = FullResolutionResidualUnit(y, z, 384, 8)
        y = layers.MaxPool2D(pool_size=[2, 2], padding="same")(y)

        y, z = FullResolutionResidualUnit(y, z, 384, 16)
        y, z = FullResolutionResidualUnit(y, z, 384, 16)
        y = layers.MaxPool2D(pool_size=[2, 2], padding="same")(y)

        y, z = FullResolutionResidualUnit(y, z, 384, 32)
        y, z = FullResolutionResidualUnit(y, z, 384, 32)

        # Upsampling Path
        y = UnpoolBilinear(2)(y)
        y, z = FullResolutionResidualUnit(y, z, 192, 16)
        y, z = FullResolutionResidualUnit(y, z, 192, 16)

        y = UnpoolBilinear(2)(y)
        y, z = FullResolutionResidualUnit(y, z, 192, 8)
        y, z = FullResolutionResidualUnit(y, z, 192, 8)

        y = UnpoolBilinear(2)(y)
        y, z = FullResolutionResidualUnit(y, z, 192, 4)
        y, z = FullResolutionResidualUnit(y, z, 192, 4)

        y = UnpoolBilinear(2)(y)
        y, z = FullResolutionResidualUnit(y, z, 96, 2)
        y, z = FullResolutionResidualUnit(y, z, 96, 2)

        y = UnpoolBilinear(2)(y)

        # Final Stage
        x = layers.Concatenate()([y, z])
        x = ResidualUnit(x, filters=48, filter_size=3)
        x = ResidualUnit(x, filters=48, filter_size=3)
        x = ResidualUnit(x, filters=48, filter_size=3)

        x = layers.Conv2D(num_classes, kernel_size=[1, 1], padding="same")(x)
        x = layers.Softmax(name="softmax_out")(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x, name="FRRN-B")

    return model
