import math

import tensorflow as tf
from keras.layers import EinsumDense


@tf.keras.utils.register_keras_serializable(package="Chambers")
class ScaledAttention(tf.keras.layers.Attention):
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
        """
        Einsum notation:
        b = batch_size
        t = sequence length
        d = embedding dimension
        n = num heads
        h = head dimension
        """

        d = input_shape[0][-1]
        t = input_shape[0][1]

        qkv_einsum_equation = "btd,dnh->bnth"
        qkv_bias_axes = "nh"
        qkv_output_shape = [self.num_heads, t, self.head_dim]  # excluding batch dimension
        self.q_dense = EinsumDense(
            qkv_einsum_equation,
            output_shape=qkv_output_shape,
            bias_axes=qkv_bias_axes,
        )
        self.v_dense = EinsumDense(
            qkv_einsum_equation,
            output_shape=qkv_output_shape,
            bias_axes=qkv_bias_axes,
        )
        self.k_dense = EinsumDense(
            qkv_einsum_equation,
            output_shape=qkv_output_shape,
            bias_axes=qkv_bias_axes,
        )

        self.proj_dense = EinsumDense(
            "bnth,ndh->btd", output_shape=[t, d], bias_axes="d"
        )

        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        q = inputs[0]  # [batch_size, tq, dim]
        v = inputs[1]  # [batch_size, tv, dim]
        k = inputs[2] if len(inputs) > 2 else v  # [batch_size, tv, dim]

        # linear projections + head split
        query = self.q_dense(q)
        value = self.v_dense(v)
        key = self.k_dense(k)

        if mask is not None:
            mask = self.separate_heads_mask(mask)

        attention = self.attention(
            [query, value, key], mask=mask, training=training
        )  # [batch_size, num_heads, tq, head_dim]

        # linear projection + head merge
        x = self.proj_dense(attention)

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
