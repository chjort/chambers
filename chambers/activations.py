from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def gelu(features, approximate=False, name=None):
    """Compute the Gaussian Error Linear Unit (GELU) activation function.
    Gaussian error linear unit (GELU) computes
    `x * P(X <= x)`, where `P(X) ~ N(0, 1)`.
    The (GELU) nonlinearity weights inputs by their value, rather than gates
    inputs by their sign as in ReLU.
    For example:
    >>> x = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=tf.float32)
    >>> y = tf.nn.gelu(x)
    >>> y.numpy()
    array([-0.00404951, -0.15865529,  0.        ,  0.8413447 ,  2.9959507 ],
        dtype=float32)
    >>> y = tf.nn.gelu(x, approximate=True)
    >>> y.numpy()
    array([-0.00363752, -0.15880796,  0.        ,  0.841192  ,  2.9963627 ],
        dtype=float32)
    Args:
      features: A `Tensor` representing preactivation values.
      approximate: An optional `bool`. Defaults to `False`. Whether to enable
        approximation.
      name: A name for the operation (optional).
    Returns:
      A `Tensor` with the same type as `features`.
    References:
      [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415).
    """
    with ops.name_scope(name, "Gelu", [features]):
        features = ops.convert_to_tensor(features, name="features")
        if approximate:
            coeff = math_ops.cast(0.044715, features.dtype)
            return (
                0.5
                * features
                * (
                    1.0
                    + math_ops.tanh(
                        0.7978845608028654
                        * (features + coeff * math_ops.pow(features, 3))
                    )
                )
            )
        else:
            return (
                0.5
                * features
                * (
                    1.0
                    + math_ops.erf(
                        features / math_ops.cast(1.4142135623730951, features.dtype)
                    )
                )
            )
