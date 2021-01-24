import tensorflow as tf


class ReduceFunctionWrapper(tf.keras.layers.Layer):
    def __init__(self, reduce_fn, axis=None, name=None, **kwargs):
        self.reduce_fn = reduce_fn
        self.axis = axis
        self.kwargs = kwargs
        super().__init__(name=name)

    def compute_output_shape(self, input_shape):
        if self.axis is None:
            return []

        return [d for idx, d in enumerate(input_shape) if idx not in self.axis]

    def call(self, inputs, **kwargs):
        x = self.reduce_fn(inputs, axis=self.axis, **self.kwargs)
        return x

    def get_config(self):
        config = {"axis": self.axis}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()) + list(self.kwargs.items()))


class Sum(ReduceFunctionWrapper):
    def __init__(self, axis=None, keepdims=False, name=None):
        super(Sum, self).__init__(
            reduce_fn=tf.reduce_sum, axis=axis, keepdims=keepdims, name=name
        )


class Prod(ReduceFunctionWrapper):
    def __init__(self, axis=None, keepdims=False, name=None):
        super(Prod, self).__init__(
            reduce_fn=tf.reduce_prod, axis=axis, keepdims=keepdims, name=name
        )


class Max(ReduceFunctionWrapper):
    def __init__(self, axis=None, keepdims=False, name=None):
        super(Max, self).__init__(
            reduce_fn=tf.reduce_max, axis=axis, keepdims=keepdims, name=name
        )


class Min(ReduceFunctionWrapper):
    def __init__(self, axis=None, keepdims=False, name=None):
        super(Min, self).__init__(
            reduce_fn=tf.reduce_min, axis=axis, keepdims=keepdims, name=name
        )


class Argmax(ReduceFunctionWrapper):
    def __init__(self, axis=None, output_type=tf.int64, name=None):
        super(Argmax, self).__init__(
            reduce_fn=tf.argmax, axis=axis, output_type=output_type, name=name
        )


class Argmin(ReduceFunctionWrapper):
    def __init__(self, axis=None, output_type=tf.int64, name=None):
        super(Argmin, self).__init__(
            reduce_fn=tf.argmin, axis=axis, output_type=output_type, name=name
        )
