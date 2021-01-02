import tensorflow as tf

from chambers.layers.attention import MultiHeadAttention


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim=512, num_heads=8, ff_dim=2048, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.multi_head_attention = MultiHeadAttention(
            embed_dim, num_heads, dropout_rate
        )
        self.dropout_attention = tf.keras.layers.Dropout(dropout_rate)
        self.add_attention = tf.keras.layers.Add()
        self.layer_norm_attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dense1 = tf.keras.layers.Dense(ff_dim, activation="relu")
        self.dense2 = tf.keras.layers.Dense(embed_dim)
        self.dropout_dense = tf.keras.layers.Dropout(dropout_rate)
        self.add_dense = tf.keras.layers.Add()
        self.layer_norm_dense = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.supports_masking = True

    def call(self, inputs, mask=None, training=None):
        attention = self.multi_head_attention(
            [inputs, inputs, inputs], mask=[mask, mask]
        )
        attention = self.dropout_attention(attention, training=training)
        x = self.add_attention([inputs, attention])
        x = self.layer_norm_attention(x)

        # Feed Forward
        dense = self.dense1(x)
        dense = self.dense2(dense)
        dense = self.dropout_dense(dense, training=training)
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
            embed_dim, num_heads, dropout_rate, causal=causal
        )
        self.dropout_attention1 = tf.keras.layers.Dropout(dropout_rate)
        self.add_attention1 = tf.keras.layers.Add()
        self.layer_norm_attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.multi_head_attention2 = MultiHeadAttention(
            embed_dim, num_heads, dropout_rate
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
        if mask is not None:
            mask1 = [mask[0], mask[0]]
            mask2 = [mask[0], mask[1]]
        else:
            mask1 = None
            mask2 = None

        attention = self.multi_head_attention1([x, x, x], mask=mask1)
        attention = self.dropout_attention1(attention, training=training)
        x = self.add_attention1([x, attention])
        x = self.layer_norm_attention1(x)

        attention = self.multi_head_attention2([x, x_encoder, x_encoder], mask=mask2)
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
        dropout_rate=0.1,
        norm=False,
        **kwargs
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.norm = norm

        if norm:
            self.norm_layer = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        else:
            self.norm_layer = None
        self.supports_masking = True

        super(Encoder, self).__init__(**kwargs)

    def build(self, input_shape):
        self.layers = [
            EncoderLayer(self.embed_dim, self.num_heads, self.ff_dim, self.dropout_rate)
            for i in range(self.num_layers)
        ]

    def call(self, inputs, mask=None, **kwargs):
        x = inputs
        for layer in self.layers:
            x = layer(x, mask=mask)

        if self.norm:
            x = self.norm_layer(x)

        return x

    def get_config(self):
        config = {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
            "num_layers": self.num_layers,
            "norm": self.norm,
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
