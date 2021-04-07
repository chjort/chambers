import math
import tensorflow as tf


class Distance(tf.keras.layers.Layer):
    def __init__(self, axis=-1, keepdims=False, **kwargs):
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

    This layer takes as input a list of two vectors [a, b] and computes
    the L1 distance between a and b according to the following equation:

            l1 = |a - b|

    """

    def call(self, inputs, **kwargs):
        a, b = inputs
        x = a - b
        x = tf.abs(x)
        x = tf.reduce_sum(x, axis=self.axis, keepdims=self.keepdims)
        return x


@tf.keras.utils.register_keras_serializable(package="Chambers")
class L2Distance(Distance):
    """
    L2 distance layer. Also knows as Euclidean distance.

    This layer takes as input a list of two vectors [a, b] and computes
    the Euclidean distance between a and b according to the following equation:

            euclidean distance = sqrt((a - b) . (a - b))

    """

    def call(self, inputs, **kwargs):
        a, b = inputs
        x = a - b
        x = tf.square(x)
        x = tf.reduce_sum(x, axis=self.axis, keepdims=self.keepdims)
        x = tf.sqrt(x)
        return x


@tf.keras.utils.register_keras_serializable(package="Chambers")
class CosineSimilarity(Distance):
    """
    Cosine distance layer

    This layer takes as input a list of two vectors [a, b] and computes
    the cosine similarity between a and b according to the following equation:

            cosine similarity = (a . b) / (||a|| * ||b||)

            scaled cosine similarity = (cosine similairty + 1) / 2

    """

    def call(self, inputs, **kwargs):
        a, b = inputs
        x = self._cosine_similarity(a, b)
        return self._scale(x)

    def _cosine_similarity(self, a, b):
        a = tf.nn.l2_normalize(a, axis=self.axis)
        b = tf.nn.l2_normalize(b, axis=self.axis)
        x = a * b
        x = tf.reduce_sum(x, axis=self.axis, keepdims=self.keepdims)
        return x

    def _scale(self, cos_sim):
        return (cos_sim + 1) / 2


class AngularCosineSimilarity(CosineSimilarity):
    def _scale(self, cos_sim):
        return 1 - tf.math.acos(cos_sim) / math.pi


class CubicCosineSimilarity(CosineSimilarity):
    def _scale(self, cos_sim):
        return 0.5 + 0.25 * cos_sim + 0.25 * tf.pow(cos_sim, 3)


class SqrtCosineSimilarity(CosineSimilarity):
    def _scale(self, cos_sim):
        return 1 - tf.sqrt((1 - cos_sim) / 2)
