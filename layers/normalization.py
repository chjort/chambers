import tensorflow as tf


class L2Normalization(tf.keras.layers.Layer):
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
