import tensorflow as tf


class L1Distance(tf.keras.layers.Layer):
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
        x = tf.abs(x)
        if self.sum:
            x = tf.reduce_sum.sum(x, axis=self.axis, keepdims=self.keepdims)
        return tf.reduce_max(x, tf.keras.backend.epsilon())

    def get_config(self):
        config = {"sum": self.sum, "axis": self.axis, "keepdims": self.keepdims}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CosineDistance(tf.keras.layers.Layer):
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
            x = tf.reduce_sum(x, axis=self.axis, keepdims=self.keepdims)
        dist = 1 - x
        return tf.reduce_max(dist, tf.keras.backend.epsilon())

    def get_config(self):
        config = {"sum": self.sum, "axis": self.axis, "keepdims": self.keepdims}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class EuclideanDistance(tf.keras.layers.Layer):
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
        x = tf.square(x)
        if self.sum:
            x = tf.reduce_sum(x, axis=self.axis, keepdims=self.keepdims)
        x = tf.sqrt(x)
        return tf.reduce_max(x, tf.keras.backend.epsilon())

    def get_config(self):
        config = {"sum": self.sum, "axis": self.axis, "keepdims": self.keepdims}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
