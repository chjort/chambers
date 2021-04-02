import tensorflow as tf

from chambers.activations import gelu
from chambers.layers.attention import MultiHeadAttention


@tf.keras.utils.register_keras_serializable(package="Chambers")
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        ff_dim=2048,
        dense_kernel_initializer="glorot_uniform",
        attention_dropout_rate=0.1,
        dense_dropout_rate=0.1,
        norm_epsilon=1e-6,
        pre_norm=False,
    ):
        super(EncoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dense_kernel_initializer = dense_kernel_initializer
        self.attention_dropout_rate = attention_dropout_rate
        self.dense_dropout_rate = dense_dropout_rate
        self.norm_epsilon = norm_epsilon
        self.pre_norm = pre_norm

        self.multi_head_attention = MultiHeadAttention(
            head_dim=embed_dim // num_heads,
            num_heads=num_heads,
            dense_kernel_initializer=self.dense_kernel_initializer,
            dropout_rate=attention_dropout_rate,
            causal=False,
        )
        self.dropout1 = tf.keras.layers.Dropout(dense_dropout_rate)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=norm_epsilon)

        # mlp
        self.dense1 = tf.keras.layers.Dense(
            ff_dim, activation=gelu, kernel_initializer=dense_kernel_initializer
        )
        self.dense2 = tf.keras.layers.Dense(
            embed_dim, kernel_initializer=dense_kernel_initializer
        )
        self.dropout2 = tf.keras.layers.Dropout(dense_dropout_rate)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=norm_epsilon)

        self.supports_masking = True

    def call(self, inputs, mask=None, training=None):
        x = inputs

        if self.pre_norm:
            x = x + self._self_attn(self.norm1(x), mask, training)
            x = x + self._mlp(self.norm2(x), training)
        else:
            x = self.norm1(x + self._self_attn(x, mask, training))
            x = self.norm2(x + self._mlp(x, training))

        return x

    def _self_attn(self, q, mask=None, training=None):
        attention = self.multi_head_attention(
            [q, q, q], mask=[mask, mask], training=training
        )
        attention = self.dropout1(attention, training=training)
        return attention

    def _mlp(self, x, training=None):
        x = self.dense1(x)
        # TODO: dropout here for ViT?
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return x

    def get_config(self):
        if isinstance(self.dense_kernel_initializer, tf.keras.initializers.Initializer):
            dense_kernel_initializer = tf.keras.initializers.serialize(
                self.dense_kernel_initializer
            )
        else:
            dense_kernel_initializer = self.dense_kernel_initializer

        config = {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dense_kernel_initializer": dense_kernel_initializer,
            "attention_dropout_rate": self.attention_dropout_rate,
            "dense_dropout_rate": self.dense_dropout_rate,
            "norm_epsilon": self.norm_epsilon,
            "pre_norm": self.pre_norm,
        }
        base_config = super(EncoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def from_config(cls, config):
        if isinstance(
            config["dense_kernel_initializer"], tf.keras.initializers.Initializer
        ):
            config["dense_kernel_initializer"] = tf.keras.initializers.deserialize(
                config["dense_kernel_initializer"]
            )

        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="Chambers")
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        ff_dim=2048,
        dense_kernel_initializer="glorot_uniform",
        attention_dropout_rate=0.1,
        dense_dropout_rate=0.1,
        norm_epsilon=1e-6,
        pre_norm=False,
        causal=True,
    ):
        super(DecoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dense_kernel_initializer = dense_kernel_initializer
        self.attention_dropout_rate = attention_dropout_rate
        self.dense_dropout_rate = dense_dropout_rate
        self.norm_epsilon = norm_epsilon
        self.pre_norm = pre_norm
        self.causal = causal

        # self-attention
        self.multi_head_attention1 = MultiHeadAttention(
            head_dim=embed_dim // num_heads,
            num_heads=num_heads,
            dense_kernel_initializer=self.dense_kernel_initializer,
            dropout_rate=attention_dropout_rate,
            causal=causal,
        )
        self.dropout1 = tf.keras.layers.Dropout(dense_dropout_rate)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=norm_epsilon)

        # cross-attention
        self.multi_head_attention2 = MultiHeadAttention(
            head_dim=embed_dim // num_heads,
            num_heads=num_heads,
            dense_kernel_initializer=self.dense_kernel_initializer,
            dropout_rate=attention_dropout_rate,
            causal=False,
        )
        self.dropout2 = tf.keras.layers.Dropout(dense_dropout_rate)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=norm_epsilon)

        # mlp
        self.dense1 = tf.keras.layers.Dense(
            ff_dim, activation=gelu, kernel_initializer=dense_kernel_initializer
        )
        self.dense2 = tf.keras.layers.Dense(
            embed_dim, kernel_initializer=dense_kernel_initializer
        )
        self.dropout3 = tf.keras.layers.Dropout(dense_dropout_rate)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=norm_epsilon)

        self.supports_masking = True

    def call(self, inputs, mask=None, training=None):
        x, x_enc = inputs
        q_mask = mask[0] if mask else None
        v_mask = mask[1] if mask else None

        if self.pre_norm:
            x = x + self._self_attn(self.norm1(x), q_mask, training)
            x = x + self._cross_attn(
                self.norm2(x),
                self.norm2(x_enc),
                q_mask,
                v_mask,
                training,
            )
            x = x + self._mlp(self.norm3(x), training)
        else:
            x = self.norm1(x + self._self_attn(x, q_mask, training))
            x = self.norm2(x + self._cross_attn(x, x_enc, q_mask, v_mask, training))
            x = self.norm3(x + self._mlp(x, training))
        return x

    def _self_attn(self, q, mask=None, training=None):
        attention = self.multi_head_attention1(
            [q, q, q], mask=[mask, mask], training=training
        )
        attention = self.dropout1(attention, training=training)
        return attention

    def _cross_attn(self, q, v, q_mask=None, v_mask=None, training=None):
        attention = self.multi_head_attention2(
            [q, v, v], mask=[q_mask, v_mask], training=training
        )
        attention = self.dropout2(attention, training=training)
        return attention

    def _mlp(self, x, training=None):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout3(x, training=training)
        return x

    def compute_mask(self, inputs, mask=None):
        if mask:
            target_mask = mask[0]
            if target_mask is None:
                return None
            return tf.convert_to_tensor(target_mask)
        return None

    def get_config(self):
        if isinstance(self.dense_kernel_initializer, tf.keras.initializers.Initializer):
            dense_kernel_initializer = tf.keras.initializers.serialize(
                self.dense_kernel_initializer
            )
        else:
            dense_kernel_initializer = self.dense_kernel_initializer

        config = {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dense_kernel_initializer": dense_kernel_initializer,
            "attention_dropout_rate": self.attention_dropout_rate,
            "dense_dropout_rate": self.dense_dropout_rate,
            "norm_epsilon": self.norm_epsilon,
            "pre_norm": self.pre_norm,
            "causal": self.causal,
        }
        base_config = super(DecoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def from_config(cls, config):
        if isinstance(
            config["dense_kernel_initializer"], tf.keras.initializers.Initializer
        ):
            config["dense_kernel_initializer"] = tf.keras.initializers.deserialize(
                config["dense_kernel_initializer"]
            )

        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="Chambers")
class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        num_layers,
        dense_kernel_initializer="glorot_uniform",
        attention_dropout_rate=0.1,
        dense_dropout_rate=0.1,
        norm_epsilon=1e-6,
        pre_norm=False,
        norm_output=False,
        **kwargs
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dense_kernel_initializer = dense_kernel_initializer
        self.attention_dropout_rate = attention_dropout_rate
        self.dense_dropout_rate = dense_dropout_rate
        self.norm_epsilon = norm_epsilon
        self.pre_norm = pre_norm
        self.norm_output = norm_output

        if norm_output:
            self.norm_layer = tf.keras.layers.LayerNormalization(epsilon=norm_epsilon)
        else:
            self.norm_layer = None
        self.supports_masking = True

        super(Encoder, self).__init__(**kwargs)

    def build(self, input_shape):
        self.layers = [
            EncoderLayer(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dense_kernel_initializer=self.dense_kernel_initializer,
                attention_dropout_rate=self.attention_dropout_rate,
                dense_dropout_rate=self.dense_dropout_rate,
                norm_epsilon=self.norm_epsilon,
                pre_norm=self.pre_norm,
            )
            for i in range(self.num_layers)
        ]
        super(Encoder, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):
        x = inputs
        for layer in self.layers:
            x = layer(x, mask=mask, training=training)

        if self.norm_output:
            x = self.norm_layer(x)

        return x

    def get_config(self):
        if isinstance(self.dense_kernel_initializer, tf.keras.initializers.Initializer):
            dense_kernel_initializer = tf.keras.initializers.serialize(
                self.dense_kernel_initializer
            )
        else:
            dense_kernel_initializer = self.dense_kernel_initializer

        config = {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_layers": self.num_layers,
            "dense_kernel_initializer": dense_kernel_initializer,
            "attention_dropout_rate": self.attention_dropout_rate,
            "dense_dropout_rate": self.dense_dropout_rate,
            "norm_epsilon": self.norm_epsilon,
            "pre_norm": self.pre_norm,
            "norm_output": self.norm_output,
        }
        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def from_config(cls, config):
        if isinstance(
            config["dense_kernel_initializer"], tf.keras.initializers.Initializer
        ):
            config["dense_kernel_initializer"] = tf.keras.initializers.deserialize(
                config["dense_kernel_initializer"]
            )

        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="Chambers")
class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        num_layers,
        dense_kernel_initializer="glorot_uniform",
        attention_dropout_rate=0.1,
        dense_dropout_rate=0.1,
        norm_epsilon=1e-6,
        pre_norm=False,
        norm_output=False,
        causal=True,
        return_sequence=False,
        **kwargs
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dense_kernel_initializer = dense_kernel_initializer
        self.attention_dropout_rate = attention_dropout_rate
        self.dense_dropout_rate = dense_dropout_rate
        self.norm_epsilon = norm_epsilon
        self.pre_norm = pre_norm
        self.norm_output = norm_output
        self.causal = causal

        if norm_output:
            self.norm_layer = tf.keras.layers.LayerNormalization(epsilon=norm_epsilon)
        else:
            self.norm_layer = None
        self.return_sequence = return_sequence
        self.supports_masking = True

        super(Decoder, self).__init__(**kwargs)

    def build(self, input_shape):
        self.layers = [
            DecoderLayer(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dense_kernel_initializer=self.dense_kernel_initializer,
                attention_dropout_rate=self.attention_dropout_rate,
                dense_dropout_rate=self.dense_dropout_rate,
                norm_epsilon=self.norm_epsilon,
                pre_norm=self.pre_norm,
                causal=self.causal,
            )
            for i in range(self.num_layers)
        ]

    def call(self, inputs, mask=None, training=None, **kwargs):
        x, x_encoder = inputs

        decode_sequence = []
        for layer in self.layers:
            x = layer([x, x_encoder], mask=mask, training=training)
            decode_sequence.append(x)

        if self.return_sequence:
            if self.norm_output:
                decode_sequence = [self.norm_layer(x) for x in decode_sequence]

            x = tf.stack(decode_sequence, axis=0)
            x = tf.transpose(x, [1, 0, 2, 3])
        else:
            x = decode_sequence[-1]
            if self.norm_output:
                x = self.norm_layer(x)

        return x

    def compute_mask(self, inputs, mask=None):
        if mask:
            target_mask = mask[0]
            if target_mask is None:
                return None
            return tf.convert_to_tensor(target_mask)
        return None

    def get_config(self):
        if isinstance(self.dense_kernel_initializer, tf.keras.initializers.Initializer):
            dense_kernel_initializer = tf.keras.initializers.serialize(
                self.dense_kernel_initializer
            )
        else:
            dense_kernel_initializer = self.dense_kernel_initializer

        config = {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_layers": self.num_layers,
            "dense_kernel_initializer": dense_kernel_initializer,
            "attention_dropout_rate": self.attention_dropout_rate,
            "dense_dropout_rate": self.dense_dropout_rate,
            "norm_epsilon": self.norm_epsilon,
            "pre_norm": self.pre_norm,
            "norm_output": self.norm_output,
            "causal": self.causal,
            "return_sequence": self.return_sequence,
        }
        base_config = super(EncoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def from_config(cls, config):
        if isinstance(
            config["dense_kernel_initializer"], tf.keras.initializers.Initializer
        ):
            config["dense_kernel_initializer"] = tf.keras.initializers.deserialize(
                config["dense_kernel_initializer"]
            )

        return cls(**config)
