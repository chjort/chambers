import math

import tensorflow as tf
from packaging import version
from tensorflow.python.keras.utils.tf_utils import smart_cond
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


@tf.keras.utils.register_keras_serializable(package="Chambers")
class Attention(tf.keras.layers.Attention):
    def _apply_scores(self, scores, value, scores_mask=None, training=None):
        """Applies attention scores to the given value tensor.

        To use this method in your attention layer, follow the steps:

        * Use `query` tensor of shape `[batch_size, Tq]` and `key` tensor of shape
          `[batch_size, Tv]` to calculate the attention `scores`.
        * Pass `scores` and `value` tensors to this method. The method applies
          `scores_mask`, calculates `attention_distribution = softmax(scores)`, then
          returns `matmul(attention_distribution, value).
        * Apply `query_mask` and return the result.

        Args:
          scores: Scores float tensor of shape `[batch_size, Tq, Tv]`.
          value: Value tensor of shape `[batch_size, Tv, dim]`.
          scores_mask: A boolean mask `Tensor` of shape `[batch_size, 1, Tv]` or
            `[batch_size, Tq, Tv]`. If given, scores at positions where
            `scores_mask==False` do not contribute to the result. It must contain
            at least one `True` value in each line along the last dimension.
          training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (no dropout).

        Returns:
          Tensor of shape `[batch_size, Tq, dim]`.
        """
        if scores_mask is not None:
            padding_mask = math_ops.logical_not(scores_mask)
            # Bias so padding positions do not contribute to attention distribution.
            scores -= tf.minimum(scores.dtype.max, 1.0e9) * math_ops.cast(
                padding_mask, dtype=scores.dtype
            )  # compatible with mixed precision
        if training is None:
            training = tf.keras.backend.learning_phase()
        weights = tf.nn.softmax(scores)

        def dropped_weights():
            return tf.nn.dropout(weights, rate=self.dropout)

        weights = smart_cond(
            training, dropped_weights, lambda: array_ops.identity(weights)
        )

        if version.parse(tf.__version__) < version.parse("2.4"):
            return math_ops.matmul(weights, value)
        else:
            return math_ops.matmul(weights, value), weights


@tf.keras.utils.register_keras_serializable(package="Chambers")
class ScaledAttention(Attention):
    def __init__(self, key_dim=None, *args, **kwargs):
        super(ScaledAttention, self).__init__(*args, **kwargs)
        self.key_dim = key_dim
        self._scale = math.sqrt(key_dim) if key_dim is not None else None

    def _calculate_scores(self, query, key):
        scores = super(ScaledAttention, self)._calculate_scores(query, key)

        if self.key_dim is None:
            key_dim = tf.cast(tf.shape(key)[-1], scores.dtype)
            scale = tf.sqrt(key_dim)
        else:
            scale = tf.cast(self._scale, scores.dtype)

        scores = scores / scale
        return scores


@tf.keras.utils.register_keras_serializable(package="Chambers")
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        head_dim=64,
        num_heads=8,
        dense_kernel_initializer="glorot_uniform",
        dropout_rate=0.1,
        causal=False,
        **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.dense_kernel_initializer = dense_kernel_initializer
        self.dropout_rate = dropout_rate
        self.causal = causal

        self.attention = ScaledAttention(
            key_dim=head_dim, causal=causal, dropout=dropout_rate
        )

        self.reshape_split_mask = tf.keras.layers.Reshape((-1, 1))
        self.permute_mask = tf.keras.layers.Permute((2, 1))

    def build(self, input_shape):
        d = input_shape[0][-1]

        self.w_query = self.add_weight(
            name="w_query",
            shape=(d, self.num_heads, self.head_dim),
            initializer=self.dense_kernel_initializer,
        )
        self.b_query = self.add_weight(
            name="b_query",
            shape=(self.num_heads, 1, self.head_dim),
            initializer="zeros",
        )

        self.w_value = self.add_weight(
            name="w_value",
            shape=(d, self.num_heads, self.head_dim),
            initializer=self.dense_kernel_initializer,
        )
        self.b_value = self.add_weight(
            name="b_value",
            shape=(self.num_heads, 1, self.head_dim),
            initializer="zeros",
        )

        self.w_key = self.add_weight(
            name="w_key",
            shape=(d, self.num_heads, self.head_dim),
            initializer=self.dense_kernel_initializer,
        )
        self.b_key = self.add_weight(
            name="b_key",
            shape=(self.num_heads, 1, self.head_dim),
            initializer="zeros",
        )

        self.w_projection = self.add_weight(
            name="w_projection",
            shape=(self.num_heads, d, self.head_dim),
            initializer=self.dense_kernel_initializer,
        )
        self.b_projection = self.add_weight(
            name="b_projection",
            shape=(1, d),
            initializer="zeros",
        )
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        """
        Einsum notation:
        b = batch_size
        t = sequence length
        d = embedding dimension
        n = num heads
        h = head dimension
        """
        q = inputs[0]  # [batch_size, tq, dim]
        v = inputs[1]  # [batch_size, tv, dim]
        k = inputs[2] if len(inputs) > 2 else v  # [batch_size, tv, dim]

        # linear projections + head split
        query = tf.einsum("btd,dnh->bnth", q, self.w_query) + self.b_query
        value = tf.einsum("btd,dnh->bnth", v, self.w_value) + self.b_value
        key = tf.einsum("btd,dnh->bnth", k, self.w_key) + self.b_key

        if mask is not None:
            mask = self.separate_heads_mask(mask)

        attention = self.attention(
            [query, value, key], mask=mask, training=training
        )  # [batch_size, num_heads, tq, head_dim]

        # linear projection + head merge
        x = tf.einsum("bnth,ndh->btd", attention, self.w_projection) + self.b_projection

        return x

    def separate_heads_mask(self, mask):
        query_mask = mask[0]  # [batch_size, tq]
        value_mask = mask[1]  # [batch_size, tv]

        if query_mask is not None:
            query_mask = self.reshape_split_mask(
                query_mask
            )  # [batch_size, tq, num_heads]
            query_mask = self.permute_mask(query_mask)  # [batch_size, num_heads, tq]

        if value_mask is not None:
            value_mask = self.reshape_split_mask(
                value_mask
            )  # [batch_size, tv, num_heads]
            value_mask = self.permute_mask(value_mask)  # [batch_size, num_heads, tv]

        return [query_mask, value_mask]

    def compute_mask(self, inputs, mask=None):
        if mask:
            q_mask = mask[0]
            if q_mask is None:
                return None
            return tf.convert_to_tensor(q_mask)
        return None

    def get_config(self):
        if isinstance(self.dense_kernel_initializer, tf.keras.initializers.Initializer):
            dense_kernel_initializer = tf.keras.initializers.serialize(
                self.dense_kernel_initializer
            )
        else:
            dense_kernel_initializer = self.dense_kernel_initializer

        config = {
            "head_dim": self.head_dim,
            "num_heads": self.num_heads,
            "dense_kernel_initializer": dense_kernel_initializer,
            "dropout_rate": self.dropout_rate,
            "causal": self.causal,
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if isinstance(
            config["dense_kernel_initializer"], tf.keras.initializers.Initializer
        ):
            config["dense_kernel_initializer"] = tf.keras.initializers.deserialize(
                config["dense_kernel_initializer"]
            )
        return cls(**config)
