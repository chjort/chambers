import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.python.keras.layers.pooling import GlobalPooling2D


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
        x = tf.pow(inputs, self.p)
        if self.data_format == 'channels_last':
            x = tf.reduce_mean(x, axis=[1, 2])
        else:
            x = tf.reduce_mean(x, axis=[2, 3])
        x = tf.pow(x, 1 / self.p)

        return x

    def get_config(self):
        config = {'p': self.p, 'trainable': self.trainable}
        base_config = super(GlobalGeneralizedMean, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
