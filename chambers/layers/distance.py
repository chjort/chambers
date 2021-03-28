import tensorflow as tf


class Distance(tf.keras.layers.Layer):
    def __init__(self, axis=-1, keepdims=True, **kwargs):
        super(Distance, self).__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def get_config(self):
        config = {"axis": self.axis, "keepdims": self.keepdims}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="Chambers")
class L1Distance(Distance):
    """
    L1 distance or "Manhattan-distance" layer

    This layer takes as input a list of two vectors [v1, v2] and computes
    the L1 distance between v1 and v2 according to the following equation:

            l1 = |v1 - v2|

    """

    def call(self, inputs, **kwargs):
        v1, v2 = inputs
        x = v1 - v2
        x = tf.abs(x)
        x = tf.reduce_sum(x, axis=self.axis, keepdims=self.keepdims)
        return x


@tf.keras.utils.register_keras_serializable(package="Chambers")
class CosineDistance(Distance):
    """
    Cosine distance layer

    This layer takes as input a list of two vectors [v1, v2] and computes
    the Cosine distance between v1 and v2 according to the following equation:

            cosine similarity = (v1 . v2) / (||v1|| * ||v2||)

            cosine distance = 1 - cosine similarity

    """

    def call(self, inputs, **kwargs):
        v1, v2 = inputs
        v1 = tf.nn.l2_normalize(v1, axis=self.axis)
        v2 = tf.nn.l2_normalize(v2, axis=self.axis)
        x = v1 * v2
        x = tf.reduce_sum(x, axis=self.axis, keepdims=self.keepdims)
        x = 1 - x
        return x


@tf.keras.utils.register_keras_serializable(package="Chambers")
class L2Distance(Distance):
    """
    L2 distance layer. Also knows as Euclidean distance.

    This layer takes as input a list of two vectors [v1, v2] and computes
    the Euclidean distance between v1 and v2 according to the following equation:

            euclidean distance = sqrt((v1 - v2) . (v1 - v2))

    """

    def call(self, inputs, **kwargs):
        v1, v2 = inputs
        x = v1 - v2
        x = tf.square(x)
        x = tf.reduce_sum(x, axis=self.axis, keepdims=self.keepdims)
        x = tf.sqrt(x)
        return x
