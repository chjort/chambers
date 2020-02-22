import tensorflow as tf


class ConvBlock(tf.keras.layers.Layer):
    """
    A convolutional block layer

    This layer layer implements a convolutional block comprised of four layers: Conv2D, Batch Normalization,
    ReLu activation, and Max-pooling.

    """

    def __init__(self, filters, conv_size, conv_strides=1, conv_padding="same", pool_size=(2, 2), pool_strides=None,
                 pool_padding="valid", name=None):
        super(ConvBlock, self).__init__(name=name)
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=conv_size,
                                           padding=conv_padding, strides=conv_strides)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=pool_size, padding=pool_padding, strides=pool_strides)

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
