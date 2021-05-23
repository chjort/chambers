import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Chambers")
class Matmul(tf.keras.layers.Layer):
    def __init__(
        self,
        transpose_a=False,
        transpose_b=False,
        adjoint_a=False,
        adjoint_b=False,
        a_is_sparse=False,
        b_is_sparse=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self.adjoint_a = adjoint_a
        self.adjoint_b = adjoint_b
        self.a_is_sparse = a_is_sparse
        self.b_is_sparse = b_is_sparse

    def compute_output_shape(self, input_shape):
        super(Matmul, self).compute_output_shape(input_shape)

    def call(self, inputs, **kwargs):
        a, b = inputs
        x = tf.matmul(
            a=a,
            b=b,
            transpose_a=self.transpose_a,
            transpose_b=self.transpose_b,
            adjoint_a=self.adjoint_a,
            adjoint_b=self.adjoint_b,
            a_is_sparse=self.a_is_sparse,
            b_is_sparse=self.b_is_sparse,
        )
        return x

    def get_config(self):
        config = {
            "transpose_a": self.transpose_a,
            "transpose_b": self.transpose_b,
            "adjoint_a": self.adjoint_a,
            "adjoint_b": self.adjoint_b,
            "a_is_sparse": self.a_is_sparse,
            "b_is_sparse": self.b_is_sparse,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


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
