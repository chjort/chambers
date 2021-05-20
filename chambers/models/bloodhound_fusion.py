import tensorflow as tf
from tensorflow.python.keras.utils import layer_utils

from chambers.activations import gelu
from chambers.layers.distance import CosineSimilarity
from chambers.layers.embedding import PositionalEmbedding2D
from chambers.models import backbones
from chambers.models.backbones.vision_transformer import _obtain_inputs
from chambers.models.bloodhound import _Pool4DAxis2, ExpandDims, MultiHeadAttention4D


@tf.keras.utils.register_keras_serializable(package="Chambers")
class ECA(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        dense_kernel_initializer="glorot_uniform",
        attention_dropout_rate=0.1,
        dense_dropout_rate=0.1,
        norm_epsilon=1e-6,
        **kwargs,
    ):
        super(ECA, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dense_kernel_initializer = dense_kernel_initializer
        self.attention_dropout_rate = attention_dropout_rate
        self.dense_dropout_rate = dense_dropout_rate
        self.norm_epsilon = norm_epsilon

        self.multi_head_attention = MultiHeadAttention4D(
            head_dim=self.embed_dim // self.num_heads,
            num_heads=self.num_heads,
            dense_kernel_initializer=self.dense_kernel_initializer,
            dropout_rate=self.attention_dropout_rate,
            causal=False,
        )
        self.dropout = tf.keras.layers.Dropout(dense_dropout_rate)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=norm_epsilon)

    def call(self, inputs, training=None, **kwargs):
        x = self.multi_head_attention([inputs, inputs, inputs], training=training)
        x = self.dropout(x, training=training)
        x = self.norm(inputs + x)
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
            "dense_kernel_initializer": dense_kernel_initializer,
            "attention_dropout_rate": self.attention_dropout_rate,
            "dense_dropout_rate": self.dense_dropout_rate,
            "norm_epsilon": self.norm_epsilon,
        }
        base_config = super(ECA, self).get_config()
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


class CFA(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        ff_dim=2048,
        dense_kernel_initializer="glorot_uniform",
        attention_dropout_rate=0.1,
        dense_dropout_rate=0.1,
        norm_epsilon=1e-6,
        **kwargs,
    ):
        super(CFA, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dense_kernel_initializer = dense_kernel_initializer
        self.attention_dropout_rate = attention_dropout_rate
        self.dense_dropout_rate = dense_dropout_rate
        self.norm_epsilon = norm_epsilon

        self.multi_head_attention = MultiHeadAttention4D(
            head_dim=self.embed_dim // self.num_heads,
            num_heads=self.num_heads,
            dense_kernel_initializer=self.dense_kernel_initializer,
            dropout_rate=self.attention_dropout_rate,
            causal=False,
        )
        self.dropout = tf.keras.layers.Dropout(dense_dropout_rate)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=norm_epsilon)

        # mlp
        self.dense1 = tf.keras.layers.Dense(
            ff_dim, activation=gelu, kernel_initializer=dense_kernel_initializer
        )
        self.dense2 = tf.keras.layers.Dense(
            embed_dim, kernel_initializer=dense_kernel_initializer
        )
        self.dropout2 = tf.keras.layers.Dropout(dense_dropout_rate)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=norm_epsilon)

    def call(self, inputs, training=None, **kwargs):
        x, x_kv = inputs
        attention = self.multi_head_attention([x, x_kv, x_kv], training=training)
        attention = self.dropout(attention, training=training)
        x = self.norm(x + attention)

        xd = self.dense1(x)
        xd = self.dense2(xd)
        xd = self.dropout2(xd)
        x = self.norm2(xd + x)
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
        }
        base_config = super(CFA, self).get_config()
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


@tf.keras.utils.register_keras_serializable(package="Chambers")
class FeatureFusionLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        ff_dim=2048,
        dense_kernel_initializer="glorot_uniform",
        attention_dropout_rate=0.1,
        dense_dropout_rate=0.1,
        norm_epsilon=1e-6,
        **kwargs,
    ):
        super(FeatureFusionLayer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dense_kernel_initializer = dense_kernel_initializer
        self.attention_dropout_rate = attention_dropout_rate
        self.dense_dropout_rate = dense_dropout_rate
        self.norm_epsilon = norm_epsilon

        self.eca1 = ECA(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dense_kernel_initializer=self.dense_kernel_initializer,
            attention_dropout_rate=self.attention_dropout_rate,
            dense_dropout_rate=dense_dropout_rate,
        )
        self.eca2 = ECA(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dense_kernel_initializer=self.dense_kernel_initializer,
            attention_dropout_rate=self.attention_dropout_rate,
            dense_dropout_rate=dense_dropout_rate,
        )
        self.cfa1 = CFA(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=ff_dim,
            dense_kernel_initializer=self.dense_kernel_initializer,
            attention_dropout_rate=self.attention_dropout_rate,
            dense_dropout_rate=dense_dropout_rate,
        )
        self.cfa2 = CFA(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=ff_dim,
            dense_kernel_initializer=self.dense_kernel_initializer,
            attention_dropout_rate=self.attention_dropout_rate,
            dense_dropout_rate=dense_dropout_rate,
        )

    def call(self, inputs, training=None, **kwargs):
        xq, xc = inputs

        # TODO: Positional embedding

        eca_q = self.eca1(xq, training=training)
        eca_c = self.eca2(xc, training=training)

        xq = self.cfa1([eca_q, eca_c])
        xc = self.cfa2([eca_c, eca_q])

        return xq, xc

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
        }
        base_config = super(FeatureFusionLayer, self).get_config()
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


class FeatureFusionDecoder(tf.keras.layers.Layer):
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
        **kwargs,
    ):
        super(FeatureFusionDecoder, self).__init__(**kwargs)
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

    def build(self, input_shape):
        self.layers = [
            FeatureFusionLayer(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                attention_dropout_rate=self.attention_dropout_rate,
                dense_dropout_rate=self.dense_dropout_rate,
            )
            for i in range(self.num_layers)
        ]
        super(FeatureFusionDecoder, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):
        x, x_kv = inputs

        for layer in self.layers:
            x, x_kv = layer([x, x_kv], mask=mask, training=training)

        if self.norm_output:
            x = self.norm_layer(x)
            x_kv = self.norm_layer(x_kv)

        return x, x_kv

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
            "norm_output": self.norm_output,
        }
        base_config = super(FeatureFusionDecoder, self).get_config()
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


def BloodhoundFunctional(
    query_shape,
    candidates_shape,
    n_layers,
    n_heads,
    ff_dim,
    dropout_rate=0.1,
    include_top=True,
    pooling=None,
    name=None,
):
    """
        q -> encoder -> -----> zq
                      |
                      v
    c -> encoder -> decoder -> zc
    """

    inputs_q = tf.keras.layers.Input(shape=query_shape)
    inputs_c = tf.keras.layers.Input(shape=candidates_shape)

    # NOTE: positional embedding here if encoder is not a transformer
    q = ExpandDims(axis=1, name="q_expand")(inputs_q)
    c = ExpandDims(axis=0, name="c_expand")(inputs_c)
    q, c = FeatureFusionDecoder(
        embed_dim=c.shape[-1],
        num_heads=n_heads,
        ff_dim=ff_dim,
        num_layers=n_layers,
        attention_dropout_rate=dropout_rate,
        dense_dropout_rate=dropout_rate,
        norm_output=False,
        name="decoder",
    )([q, c])

    if pooling is not None:
        q = _Pool4DAxis2(method=pooling, name="pool_q")(q)
        c = _Pool4DAxis2(method=pooling, name="pool_c")(c)

    if include_top:
        if pooling is None:
            raise ValueError(
                "`include_top=True` requires `pooling` to be either 'avg', 'max', 'sum', or 'cls'."
            )

        x = CosineSimilarity(axis=-1)([q, c])
        # x = tf.keras.layers.Dense(1, activation="sigmoid", dtype=tf.float32)(c)
        # x = tf.keras.layers.Lambda(lambda x: x[:, :, 0])(x)
    else:
        x = [q, c]

    model = tf.keras.Model(inputs=[inputs_q, inputs_c], outputs=x, name=name)
    return model


def Bloodhound4D(
    n_layers,
    n_heads,
    ff_dim,
    dropout_rate=0.1,
    query_tensor=None,
    query_shape=None,
    candidates_tensor=None,
    candidates_shape=None,
    include_top=True,
    weights="imagenet21k+_224",
    pooling=None,
    model_name=None,
):
    """
        q -> encoder -> -----> zq
                      |
                      v
    c -> encoder -> decoder -> zc
    """
    enc = backbones.ViTB16(
        weights="imagenet21k+_224",
        pooling=None,
        include_top=False,
    )
    patch_size = enc.get_layer("patch_embeddings").get_layer("embedding").kernel_size
    enc = tf.keras.Model(enc.inputs, enc.outputs, name="encoder")

    inputs_q = _obtain_inputs(
        query_tensor,
        query_shape,
        default_size=224,
        min_size=patch_size,
        weights=weights,
        model_name=model_name,
        name="query",
    )
    inputs_c = _obtain_inputs(
        candidates_tensor,
        candidates_shape,
        default_size=224,
        min_size=patch_size,
        weights=weights,
        model_name=model_name,
        name="candidates",
    )

    q_enc = enc(inputs_q)
    c_enc = enc(inputs_c)
    # NOTE: positional embedding here if encoder is not a transformer
    dec = BloodhoundFunctional(
        query_shape=q_enc.shape[1:],
        candidates_shape=c_enc.shape[1:],
        n_layers=n_layers,
        n_heads=n_heads,
        ff_dim=ff_dim,
        dropout_rate=dropout_rate,
        include_top=include_top,
        pooling=pooling,
        name="decoder",
    )
    x = dec([q_enc, c_enc])

    x = tf.keras.layers.Activation("linear", dtype=tf.float32, name="cast_float32")(x)

    if query_tensor is not None:
        inputs_q = layer_utils.get_source_inputs(query_tensor)
    if candidates_tensor is not None:
        inputs_c = layer_utils.get_source_inputs(candidates_tensor)

    model = tf.keras.Model(inputs=[inputs_q, inputs_c], outputs=x, name=model_name)
    return model


def BloodhoundRes(
    embed_dim,
    n_layers,
    n_heads,
    ff_dim,
    dropout_rate=0.1,
    query_tensor=None,
    query_shape=None,
    candidates_tensor=None,
    candidates_shape=None,
    include_top=True,
    weights="imagenet21k+_224",
    pooling=None,
    model_name=None,
):
    """
        q -> encoder -> -----> zq
                      |
                      v
    c -> encoder -> decoder -> zc
    """
    inputs_q = _obtain_inputs(
        query_tensor,
        query_shape,
        default_size=224,
        min_size=32,
        weights=weights,
        model_name=model_name,
        name="query",
    )
    inputs_c = _obtain_inputs(
        candidates_tensor,
        candidates_shape,
        default_size=224,
        min_size=32,
        weights=weights,
        model_name=model_name,
        name="candidates",
    )

    enc = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    enc = tf.keras.Model(
        inputs=enc.inputs, outputs=enc.get_layer("conv4_block6_out").output
    )
    x = enc.output
    x = tf.keras.layers.Conv2D(filters=embed_dim, kernel_size=1)(x)
    x = PositionalEmbedding2D(embedding_dim=embed_dim, add_to_input=True)(x)
    x = tf.keras.layers.Reshape([x.shape[1] * x.shape[2], x.shape[3]])(x)
    enc = tf.keras.Model(inputs=enc.inputs, outputs=x, name="encoder")

    q = enc(inputs_q)
    c = enc(inputs_c)

    # NOTE: positional embedding here if encoder is not a transformer
    dec = BloodhoundFunctional(
        query_shape=q.shape[1:],
        candidates_shape=c.shape[1:],
        n_layers=n_layers,
        n_heads=n_heads,
        ff_dim=ff_dim,
        dropout_rate=dropout_rate,
        include_top=include_top,
        pooling=pooling,
        name="decoder",
    )
    x = dec([q, c])

    x = tf.keras.layers.Activation("linear", dtype=tf.float32, name="cast_float32")(x)

    if query_tensor is not None:
        inputs_q = layer_utils.get_source_inputs(query_tensor)
    if candidates_tensor is not None:
        inputs_c = layer_utils.get_source_inputs(candidates_tensor)

    model = tf.keras.Model(inputs=[inputs_q, inputs_c], outputs=x, name=model_name)
    return model
