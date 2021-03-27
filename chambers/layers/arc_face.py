import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Chambers")
class ArcFaceHead(tf.keras.layers.Layer):
    def __init__(self, n_classes, name=None, **kwargs):
        super(ArcFaceHead, self).__init__(name=name, **kwargs)
        self.n_classes = n_classes

    def build(self, input_shape):
        embedding_dim = input_shape[-1]
        self.embedding_weights = self.add_weight("embedding_weights", shape=[embedding_dim, self.n_classes])

    def call(self, inputs, **kwargs):
        x_norm = tf.nn.l2_normalize(inputs, axis=1)
        w_norm = tf.nn.l2_normalize(self.embedding_weights, axis=0)

        cos_t = tf.matmul(x_norm, w_norm)
        return cos_t

    def get_config(self):
        config = {"n_classes": self.n_classes}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))