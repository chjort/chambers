import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, BatchNormalization, ReLU
from tensorflow.keras.layers.pooling import GlobalPooling2D


class L1Distance(Layer):
    """
    L1 distance or "Manhattan-distance" layer

    This layer takes as input a list of two vectors [v1, v2] and computes
    the L1 distance between v1 and v2 according to the following equation:

            l1 = |v1 - v2|

    """

    def __init__(self, sum=True, axis=-1, keepdims=True):
        super(L1Distance, self).__init__()
        self.sum = sum
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs, **kwargs):
        v1, v2 = inputs
        x = v1 - v2
        x = K.abs(x)
        if self.sum:
            x = K.sum(x, axis=self.axis, keepdims=self.keepdims)
        return K.maximum(x, K.epsilon())

    def get_config(self):
        config = {"sum": self.sum, "axis": self.axis, "keepdims": self.keepdims}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def l1_distance(v1, v2, sum=True, axis=-1, keepdims=True):
    return L1Distance(sum, axis, keepdims)([v1, v2])


class CosineDistance(Layer):
    """
    Cosine distance layer

    This layer takes as input a list of two vectors [v1, v2] and computes
    the Cosine distance between v1 and v2 according to the following equation:

            cosine similarity = (v1 . v2) / (||v1|| * ||v2||)

            cosine distance = 1 - cosine similarity

    """

    def __init__(self, sum=True, axis=-1, keepdims=True):
        super(CosineDistance, self).__init__()
        self.sum = sum
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs, **kwargs):
        v1, v2 = inputs
        v1 = tf.nn.l2_normalize(v1, axis=self.axis)
        v2 = tf.nn.l2_normalize(v2, axis=self.axis)
        x = v1 * v2
        if self.sum:
            x = K.sum(x, axis=self.axis, keepdims=self.keepdims)
        dist = 1 - x
        return K.maximum(dist, K.epsilon())

    def get_config(self):
        config = {"sum": self.sum, "axis": self.axis, "keepdims": self.keepdims}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def cosine_distance(v1, v2, sum=True, axis=-1, keepdims=True):
    return CosineDistance(sum, axis, keepdims)([v1, v2])


class EuclideanDistance(Layer):
    """
    Euclidean distance layer

    This layer takes as input a list of two vectors [v1, v2] and computes
    the Euclidean distance between v1 and v2 according to the following equation:

            euclidean distance = sqrt((v1 - v2) . (v1 - v2))

    """

    def __init__(self, sum=True, axis=-1, keepdims=True):
        super(EuclideanDistance, self).__init__()
        self.sum = sum
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs, **kwargs):
        # v1, v2 = inputs
        x = tf.keras.layers.Subtract()(inputs)
        x = K.square(x)
        if self.sum:
            x = K.sum(x, axis=self.axis, keepdims=self.keepdims)
        x = K.sqrt(x)
        return K.maximum(x, K.epsilon())

    def get_config(self):
        config = {"sum": self.sum, "axis": self.axis, "keepdims": self.keepdims}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def euclidean_distance(v1, v2, sum=True, axis=-1, keepdims=True):
    return EuclideanDistance(sum, axis, keepdims)([v1, v2])


class ConvBlock(Layer):
    """
    A convolutional block layer

    This layer layer implements a convolutional block comprised of four layers: Conv2D, Batch Normalization,
    ReLu activation, and Max-pooling.

    """

    def __init__(self, filters, conv_size, conv_strides=1, conv_padding="same", pool_size=(2, 2), pool_strides=None,
                 pool_padding="valid", name=None):
        super(ConvBlock, self).__init__(name=name)
        self.conv = Conv2D(filters=filters, kernel_size=conv_size,
                           padding=conv_padding, strides=conv_strides)
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.maxpool = MaxPool2D(pool_size=pool_size, padding=pool_padding, strides=pool_strides)

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class GlobalGeneralizedMean(GlobalPooling2D):
    """
    Global Generalized Mean layer for spatial inputs

    This layer generalizes between max-pooling and average-pooling determined by the parameter p.
    The parameter p is trainable and can be learned from backpropagation.

    # References
        - [Fine-tuning CNN Image Retrieval with No Human Annotation](https://arxiv.org/abs/1711.02512)

    """

    def __init__(self, p=3, trainable=True, data_format=None, **kwargs):
        super(GlobalGeneralizedMean, self).__init__(data_format=data_format, **kwargs)
        self.p = p
        self.trainable = trainable

    def build(self, input_shape):
        self.p = self.add_weight(shape=[1],
                                 initializer=initializers.constant(self.p),
                                 trainable=self.trainable
                                 )

    def call(self, inputs, **kwargs):
        x = K.pow(inputs, self.p)
        if self.data_format == 'channels_last':
            x = K.mean(x, axis=[1, 2])
        else:
            x = K.mean(x, axis=[2, 3])
        x = K.pow(x, 1 / self.p)

        return x

    def get_config(self):
        config = {'p': self.p, 'trainable': self.trainable}
        base_config = super(GlobalGeneralizedMean, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class L2Normalization(Layer):
    """
    L2-Normalization layer

    Normalizes the input by its L2 norm

    """

    def __init__(self, axis, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        x = tf.nn.l2_normalize(inputs, axis=self.axis)
        return x

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
