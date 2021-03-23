import tensorflow as tf


class ReduceFunctionWrapper(tf.keras.layers.Layer):
    def __init__(self, reduce_fn, axis=None, keepdims=False, name=None, **kwargs):
        self.reduce_fn = reduce_fn
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(name=name, **kwargs)

    def compute_output_shape(self, input_shape):
        if self.axis is None:
            return []

        return [d for idx, d in enumerate(input_shape) if idx not in self.axis]

    def call(self, inputs, **kwargs):
        x = self.reduce_fn(inputs, axis=self.axis, keepdims=self.keepdims)
        return x

    def get_config(self):
        config = {"axis": self.axis, "keepdims": self.keepdims}
        base_config = super().get_config()
        return dict(
            list(base_config.items()) + list(config.items()) + list(self.kwargs.items())
        )


class ArgReduceFunctionWrapper(tf.keras.layers.Layer):
    def __init__(self, reduce_fn, axis=None, output_type=tf.int64, name=None, **kwargs):
        self.reduce_fn = reduce_fn
        self.axis = axis
        self.output_type = output_type
        super().__init__(name=name, **kwargs)

    def compute_output_shape(self, input_shape):
        if self.axis is None:
            return []

        return [d for idx, d in enumerate(input_shape) if idx not in self.axis]

    def call(self, inputs, **kwargs):
        x = self.reduce_fn(inputs, axis=self.axis, output_type=self.output_type)
        return x

    def get_config(self):
        config = {"axis": self.axis, "output_type": self.output_type}
        base_config = super().get_config()
        return dict(
            list(base_config.items()) + list(config.items()) + list(self.kwargs.items())
        )


@tf.keras.utils.register_keras_serializable(package="Chambers")
class Sum(ReduceFunctionWrapper):
    def __init__(self, axis=None, keepdims=False, name=None, **kwargs):
        super(Sum, self).__init__(
            reduce_fn=tf.reduce_sum, axis=axis, keepdims=keepdims, name=name, **kwargs
        )


@tf.keras.utils.register_keras_serializable(package="Chambers")
class Prod(ReduceFunctionWrapper):
    def __init__(self, axis=None, keepdims=False, name=None, **kwargs):
        super(Prod, self).__init__(
            reduce_fn=tf.reduce_prod, axis=axis, keepdims=keepdims, name=name, **kwargs
        )


@tf.keras.utils.register_keras_serializable(package="Chambers")
class Max(ReduceFunctionWrapper):
    def __init__(self, axis=None, keepdims=False, name=None, **kwargs):
        super(Max, self).__init__(
            reduce_fn=tf.reduce_max, axis=axis, keepdims=keepdims, name=name, **kwargs
        )


@tf.keras.utils.register_keras_serializable(package="Chambers")
class Min(ReduceFunctionWrapper):
    def __init__(self, axis=None, keepdims=False, name=None, **kwargs):
        super(Min, self).__init__(
            reduce_fn=tf.reduce_min, axis=axis, keepdims=keepdims, name=name, **kwargs
        )


@tf.keras.utils.register_keras_serializable(package="Chambers")
class Argmax(ArgReduceFunctionWrapper):
    def __init__(self, axis=None, output_type=tf.int64, name=None, **kwargs):
        super(Argmax, self).__init__(
            reduce_fn=tf.argmax, axis=axis, output_type=output_type, name=name, **kwargs
        )


@tf.keras.utils.register_keras_serializable(package="Chambers")
class Argmin(ArgReduceFunctionWrapper):
    def __init__(self, axis=None, output_type=tf.int64, name=None, **kwargs):
        super(Argmin, self).__init__(
            reduce_fn=tf.argmin, axis=axis, output_type=output_type, name=name, **kwargs
        )
