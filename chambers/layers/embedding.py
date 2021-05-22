import numpy as np
import tensorflow as tf


def angle_rates(embedding_range, embedding_dim, temperature=10000.0):
    embedding_range = tf.expand_dims(tf.cast(embedding_range, tf.float32), 0)
    embedding_dim = tf.cast(embedding_dim, tf.float32)

    exponent = (2.0 * (embedding_range // 2.0)) / embedding_dim
    angle_rates = 1.0 / tf.pow(temperature, exponent)
    return angle_rates


def sequence_sin_cos_angles(seq, embedding_dim, temperature=10000.0):
    embedding_range = tf.range(embedding_dim, dtype=tf.float32)

    angles = angle_rates(embedding_range, embedding_dim, temperature)
    angle_rads = seq * angles

    # apply sin to even indices in the array; 2i
    sine_pos = tf.sin(angle_rads[..., 0::2])

    # apply cos to odd indices in the array; 2i+1
    cos_pos = tf.cos(angle_rads[..., 1::2])

    # interleave sine and cosine
    sine_cos = tf.stack([sine_pos, cos_pos], axis=-1)

    seq_len = tf.shape(seq)[0]
    pos_encoding = tf.reshape(sine_cos, [1, seq_len, -1])
    return pos_encoding


@tf.keras.utils.register_keras_serializable(package="Chambers")
class PositionalEncoding1D(tf.keras.layers.Layer):
    def __init__(self, temperature=10000, add_to_input=True, **kwargs):
        super(PositionalEncoding1D, self).__init__(**kwargs)
        self.temperature = float(temperature)
        self.add_to_input = add_to_input
        self.supports_masking = True
        self.input_spec = tf.keras.layers.InputSpec(ndim=3)

    def build(self, input_shape):
        sequence_len = input_shape[1]
        embedding_dim = input_shape[2]
        self._pos_encoding = self.positional_encoding(sequence_len, embedding_dim)
        super(PositionalEncoding1D, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        x = tf.cast(self._pos_encoding, inputs.dtype)

        if self.add_to_input:
            x = inputs + x

        return x

    def positional_encoding(self, seq_len, embedding_dim):
        seq_range = tf.expand_dims(tf.range(seq_len, dtype=tf.float32), 1)
        pos_encoding = sequence_sin_cos_angles(
            seq_range, embedding_dim, self.temperature
        )
        return pos_encoding

    def get_config(self):
        config = {
            "temperature": self.temperature,
            "add_to_input": self.add_to_input,
        }
        base_config = super(PositionalEncoding1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="Chambers")
class PositionalEncoding2D(tf.keras.layers.Layer):

    # These are the default parameters used in the original project
    def __init__(
        self,
        temperature=10000,
        normalize=False,
        scale=None,
        eps=1e-6,
        add_to_input=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        self.input_spec = tf.keras.layers.InputSpec(ndim=4)

    def build(self, input_shape):
        height = input_shape[1]
        width = input_shape[2]
        embedding_dim = input_shape[3]
        embedding_dim_1d = embedding_dim // 2
        self._pos_encoding = self.positional_encoding(height, width, embedding_dim_1d)

        super(PositionalEncoding2D, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        x = tf.cast(self._pos_encoding, inputs.dtype)

        if self.add_to_input:
            x = inputs + x

        return x

    def positional_encoding(self, height, width, embedding_dim):
        height_range = tf.reshape(tf.range(width, dtype=tf.float32), [-1, 1, 1])
        width_range = tf.reshape(tf.range(height, dtype=tf.float32), [-1, 1, 1])

        if self.normalize:
            height_range = height_range / (height_range[-1:, ...] + self.eps) * self.scale
            width_range = width_range / (width_range[-1:, ...] + self.eps) * self.scale

        sine_cos_x = sequence_sin_cos_angles(
            height_range, embedding_dim, self.temperature
        )
        sine_cos_y = sequence_sin_cos_angles(
            width_range, embedding_dim, self.temperature
        )
        sine_cos_y = tf.transpose(sine_cos_y, [1, 0, 2])

        sine_cos_x = tf.broadcast_to(sine_cos_x, [height, width, embedding_dim])
        sine_cos_y = tf.broadcast_to(sine_cos_y, [height, width, embedding_dim])

        pos_encoding = tf.concat([sine_cos_y, sine_cos_x], axis=-1)
        pos_encoding = tf.expand_dims(pos_encoding, 0)

        return pos_encoding

    def get_config(self):
        config = {
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
