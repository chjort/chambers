import tensorflow as tf

from chambers.activations import gelu
from chambers.layers.attention import MultiHeadAttention


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
    ):
        super(EncoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dense_kernel_initializer = dense_kernel_initializer
        self.attention_dropout_rate = attention_dropout_rate
        self.dense_dropout_rate = dense_dropout_rate
        self.norm_epsilon = norm_epsilon

        self.multi_head_attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dense_kernel_initializer=self.dense_kernel_initializer,
            dropout_rate=attention_dropout_rate,
            causal=False,
        )
        self.dropout1 = tf.keras.layers.Dropout(dense_dropout_rate)
        self.add_attention = tf.keras.layers.Add()
        self.layer_norm_attention = tf.keras.layers.LayerNormalization(
            epsilon=norm_epsilon
        )

        # intermediate
        self.dense1 = tf.keras.layers.Dense(
            ff_dim, activation=gelu, kernel_initializer=dense_kernel_initializer
        )

        # output
        self.dense2 = tf.keras.layers.Dense(
            embed_dim, kernel_initializer=dense_kernel_initializer
        )
        self.dropout2 = tf.keras.layers.Dropout(dense_dropout_rate)
        self.add_dense = tf.keras.layers.Add()
        self.layer_norm_dense = tf.keras.layers.LayerNormalization(epsilon=norm_epsilon)

        self.supports_masking = True

    def call(self, inputs, mask=None, training=None):
        attention = self.multi_head_attention(
            [inputs, inputs, inputs], mask=[mask, mask], training=training
        )

        # attention output
        attention = self.dropout1(attention, training=training)
        x = self.add_attention([inputs, attention])
        x = self.layer_norm_attention(x)

        # Intermediate
        dense = self.dense1(x)

        # Output
        dense = self.dense2(dense)
        dense = self.dropout2(dense, training=training)
        x = self.add_dense([x, dense])
        x = self.layer_norm_dense(x)

        return x

    def get_config(self):
        config = {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        }
        base_config = super(EncoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self, embed_dim=512, num_heads=8, ff_dim=2048, dropout_rate=0.1, causal=True
    ):
        super(DecoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.causal = causal

        self.multi_head_attention1 = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            causal=causal,
        )
        self.dropout_attention1 = tf.keras.layers.Dropout(dropout_rate)
        self.add_attention1 = tf.keras.layers.Add()
        self.layer_norm_attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.multi_head_attention2 = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        self.dropout_attention2 = tf.keras.layers.Dropout(dropout_rate)
        self.add_attention2 = tf.keras.layers.Add()
        self.layer_norm_attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dense1 = tf.keras.layers.Dense(ff_dim, activation="relu")
        self.dense2 = tf.keras.layers.Dense(embed_dim)
        self.dropout_dense = tf.keras.layers.Dropout(dropout_rate)
        self.add_dense = tf.keras.layers.Add()
        self.layer_norm_dense = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, mask=None, training=None):
        x, x_encoder = inputs
        q_mask = mask[0] if mask else None
        v_mask = mask[1] if mask else None

        attention = self.multi_head_attention1([x, x, x], mask=[q_mask, q_mask])
        attention = self.dropout_attention1(attention, training=training)
        x = self.add_attention1([x, attention])
        x = self.layer_norm_attention1(x)

        attention = self.multi_head_attention2(
            [x, x_encoder, x_encoder], mask=[q_mask, v_mask]
        )
        attention = self.dropout_attention2(attention, training=training)
        x = self.add_attention2([x, attention])
        x = self.layer_norm_attention2(x)

        # Feed Forward
        dense = self.dense1(x)
        dense = self.dense2(dense)
        dense = self.dropout_dense(dense, training=training)
        x = self.add_dense([x, dense])
        x = self.layer_norm_dense(x)

        return x

    def compute_mask(self, inputs, mask=None):
        if mask:
            target_mask = mask[0]
            if target_mask is None:
                return None
            return tf.convert_to_tensor(target_mask)
        return None

    def get_config(self):
        config = {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
            "causal": self.causal,
        }
        base_config = super(DecoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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
        config = {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
            "num_layers": self.num_layers,
            "norm": self.norm_output,
        }
        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        num_layers,
        dropout_rate=0.1,
        norm=False,
        causal=True,
        return_sequence=False,
        **kwargs
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.causal = causal

        self.norm = norm
        if norm:
            self.norm_layer = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        else:
            self.norm_layer = None
        self.return_sequence = return_sequence
        self.supports_masking = True

        super(Decoder, self).__init__(**kwargs)

    def build(self, input_shape):
        self.layers = [
            DecoderLayer(
                self.embed_dim,
                self.num_heads,
                self.ff_dim,
                self.dropout_rate,
                self.causal,
            )
            for i in range(self.num_layers)
        ]

    def call(self, inputs, mask=None, **kwargs):
        x, x_encoder = inputs

        decode_sequence = []
        for layer in self.layers:
            x = layer([x, x_encoder], mask=mask)
            decode_sequence.append(x)

        if self.norm:
            decode_sequence = [self.norm_layer(x) for x in decode_sequence]

        if self.return_sequence:
            x = tf.stack(decode_sequence, axis=0)
            x = tf.transpose(x, [1, 0, 2, 3])
        else:
            x = decode_sequence[-1]

        return x

    def compute_mask(self, inputs, mask=None):
        if mask:
            target_mask = mask[0]
            if target_mask is None:
                return None
            return tf.convert_to_tensor(target_mask)
        return None

    def get_config(self):
        config = {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
            "num_layers": self.num_layers,
            "causal": self.causal,
            "norm": self.norm,
            "return_sequence": self.return_sequence,
        }
        base_config = super(Decoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


tf.keras.utils.get_custom_objects().update(
    {
        "EncoderLayer": EncoderLayer,
        "DecoderLayer": DecoderLayer,
        "Encoder": Encoder,
        "Decoder": Decoder,
    }
)
