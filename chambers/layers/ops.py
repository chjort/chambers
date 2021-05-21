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
