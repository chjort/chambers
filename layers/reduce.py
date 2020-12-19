import tensorflow as tf


class Sum(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        if type(axis) is int:
            self.axis = (axis,)
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return [d for idx, d in enumerate(input_shape) if idx not in self.axis]

    def call(self, inputs, **kwargs):
        x = tf.reduce_sum(inputs, axis=self.axis)
        return x

    def get_config(self):
        config = {"axis": self.axis}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
