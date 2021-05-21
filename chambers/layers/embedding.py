import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Chambers")
class PositionalEncoding1D(tf.keras.layers.Layer):
    # TODO: Refactor this class to only compute embeddings once, and not every call.

    def __init__(
        self, embedding_dim=None, temperature=10000, add_to_input=True, **kwargs
    ):
        super(PositionalEncoding1D, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.temperature = float(temperature)
        self.add_to_input = add_to_input
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 3
        super(PositionalEncoding1D, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, mask=None, **kwargs):
        sequence_len = tf.shape(inputs)[1]
        if self.embedding_dim is None:
            embedding_dim = tf.shape(inputs)[-1]
        else:
            embedding_dim = self.embedding_dim

        x = self.positional_encoding(sequence_len, embedding_dim)
        x = tf.cast(x, inputs.dtype)

        if self.add_to_input:
            x = inputs + x

        return x

    def get_angles(self, seq_range, embedding_range, embedding_dim):
        seq_range = tf.expand_dims(tf.cast(seq_range, tf.float32), 1)
        embedding_range = tf.expand_dims(tf.cast(embedding_range, tf.float32), 0)
        embedding_dim = tf.cast(embedding_dim, tf.float32)

        exponent = (2.0 * (embedding_range // 2.0)) / embedding_dim
        angle_rates = 1.0 / tf.pow(self.temperature, exponent)
        return seq_range * angle_rates

    def positional_encoding(self, seq_len, embedding_dim):
        seq_range = tf.range(seq_len)
        embedding_range = tf.range(embedding_dim)

        angle_rads = self.get_angles(seq_range, embedding_range, embedding_dim)

        # apply sin to even indices in the array; 2i
        sine_pos = tf.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        cos_pos = tf.cos(angle_rads[:, 1::2])

        # interleave sine and cosine
        sine_cos = tf.stack([sine_pos, cos_pos], axis=-1)
        pos_encoding = tf.reshape(sine_cos, [1, seq_len, -1])

        return pos_encoding

    def get_config(self):
        config = {
            "embedding_dim": self.embedding_dim,
            "temperature": self.temperature,
            "add_to_input": self.add_to_input,
        }
        base_config = super(PositionalEncoding1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="Chambers")
class PositionalEncoding2D(tf.keras.layers.Layer):
    # TODO: Refactor this class to only compute embeddings once, and not every call.

    # These are the default parameters used in the original project
    def __init__(
        self,
        embedding_dim,
        temperature=10000,
        normalize=False,
        scale=None,
        eps=1e-6,
        add_to_input=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.embedding_dim_1d = embedding_dim // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * np.pi
        self.scale = scale
        self.eps = eps
        self.add_to_input = add_to_input
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, mask=None, **kwargs):
        tf.assert_rank(inputs, 4)

        if mask is not None:
            tf.assert_rank(mask, 3)
            ones = tf.cast(mask, tf.float32)
        else:
            ones = tf.ones(
                tf.shape(inputs)[:-1], dtype=tf.float32
            )  # shape [batch_size, h, w]

        x = self.compute_positional_mask(ones)

        if self.add_to_input:
            x = inputs + tf.cast(x, inputs.dtype)

        return x

    def compute_positional_mask(self, input_mask):
        y_embed = tf.math.cumsum(input_mask, axis=1)
        x_embed = tf.math.cumsum(input_mask, axis=2)

        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = tf.range(self.embedding_dim_1d, dtype=tf.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.embedding_dim_1d)

        pos_x = x_embed[..., tf.newaxis] / dim_t
        pos_y = y_embed[..., tf.newaxis] / dim_t

        pos_x = tf.stack(
            [tf.math.sin(pos_x[..., 0::2]), tf.math.cos(pos_x[..., 1::2])], axis=4
        )

        pos_y = tf.stack(
            [tf.math.sin(pos_y[..., 0::2]), tf.math.cos(pos_y[..., 1::2])], axis=4
        )

        shape = [tf.shape(pos_x)[i] for i in range(3)] + [-1]
        pos_x = tf.reshape(pos_x, shape)
        pos_y = tf.reshape(pos_y, shape)

        pos_emb = tf.concat([pos_y, pos_x], axis=3)
        return pos_emb

    def get_config(self):
        config = {
            "embedding_dim": self.embedding_dim,
            "temperature": self.temperature,
            "normalize": self.normalize,
            "scale": self.scale,
            "eps": self.eps,
            "add_to_input": self.add_to_input,
        }
        base_config = super(PositionalEncoding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="Chambers")
class LearnedEmbedding1D(tf.keras.layers.Layer):
    def __init__(
        self,
        initializer=None,
        dtype=None,
        add_to_input=True,
        name="learned_embedding",
        **kwargs,
    ):
        self.initializer = initializer
        self.add_to_input = add_to_input
        self.supports_masking = True
        super(LearnedEmbedding1D, self).__init__(dtype=dtype, name=name, **kwargs)

    def build(self, input_shape):
        self.embedding = self.add_weight(
            "embeddings",
            shape=[input_shape[1], input_shape[-1]],
            initializer=self.initializer,
            dtype=self.dtype,
        )

    def call(self, inputs, **kwargs):
        if self.add_to_input:
            return inputs + self.embedding
        else:
            return self.embedding

    def get_config(self):
        if isinstance(self.initializer, tf.keras.initializers.Initializer):
            initializer = tf.keras.initializers.serialize(self.initializer)
        else:
            initializer = self.initializer

        config = {
            "initializer": initializer,
            "add_to_input": self.add_to_input,
        }
        base_config = super(LearnedEmbedding1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if isinstance(config["initializer"], tf.keras.initializers.Initializer):
            config["initializer"] = tf.keras.initializers.deserialize(
                config["initializer"]
            )

        return cls(**config)


class LearnedEmbedding0D(LearnedEmbedding1D):
    def build(self, input_shape):
        self.embedding = self.add_weight(
            "embeddings",
            shape=[1, input_shape[-1]],
            initializer=self.initializer,
            dtype=self.dtype,
        )


@tf.keras.utils.register_keras_serializable(package="Chambers")
class ConcatEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        n_embeddings,
        embedding_dim,
        axis=-1,
        side="left",
        initializer=None,
        dtype=None,
        name="concat_embedding",
        **kwargs,
    ):
        assert (
            side == "left" or side == "right"
        ), "Argument `side` must be either 'left' or 'right'."

        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.axis = axis
        self.side = side
        self.initializer = initializer
        self.concat = tf.keras.layers.Concatenate(axis=axis)
        super(ConcatEmbedding, self).__init__(dtype=dtype, name=name, **kwargs)

    def build(self, input_shape):
        self.embedding = self.add_weight(
            "embeddings",
            shape=[self.n_embeddings, self.embedding_dim],
            initializer=self.initializer,
            dtype=self.dtype,
        )

    def call(self, inputs, **kwargs):
        batch_size = tf.shape(inputs)[0]
        embedding = tf.broadcast_to(
            self.embedding, shape=(batch_size, self.n_embeddings, self.embedding_dim)
        )

        if self.side == "left":
            x = [embedding, inputs]
        else:
            x = [inputs, embedding]

        return self.concat(x)

    def get_config(self):
        if isinstance(self.initializer, tf.keras.initializers.Initializer):
            initializer = tf.keras.initializers.serialize(self.initializer)
        else:
            initializer = self.initializer

        config = {
            "n_embeddings": self.n_embeddings,
            "embedding_dim": self.embedding_dim,
            "axis": self.axis,
            "side": self.side,
            "initializer": initializer,
        }
        base_config = super(ConcatEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if isinstance(config["initializer"], tf.keras.initializers.Initializer):
            config["initializer"] = tf.keras.initializers.deserialize(
                config["initializer"]
            )

        return cls(**config)
