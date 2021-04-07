import tensorflow as tf

from chambers.layers.reduce import Sum
from chambers.layers.transformer import Decoder
from chambers.models import backbones
from chambers.models.backbones.vision_transformer import _obtain_inputs
from chambers.layers.normalization import L2Normalization


def _pool(x, method=None, prefix=""):
    if method == "avg":
        x = tf.keras.layers.Cropping1D((1, 0), name=prefix + "sequence_embeddings")(x)
        x = tf.keras.layers.GlobalAveragePooling1D(name=prefix + "avg_pool")(x)
    elif method == "max":
        x = tf.keras.layers.Cropping1D((1, 0), name=prefix + "sequence_embeddings")(x)
        x = tf.keras.layers.GlobalMaxPooling1D(name=prefix + "max_pool")(x)
    elif method == "sum":
        x = tf.keras.layers.Cropping1D((1, 0), name=prefix + "sequence_embeddings")(x)
        x = Sum(axis=1, name=prefix + "sum_pool")(x)
    elif method == "cls":
        x = tf.keras.Sequential(
            [
                tf.keras.layers.Cropping1D((0, x.shape[1] - 1)),
                tf.keras.layers.Reshape([-1]),
            ],
            name=prefix + "cls_embedding",
        )(x)

    return x

class Matmul(tf.keras.layers.Layer):

    def compute_output_shape(self, input_shape):
        shape_a, shape_b = input_shape
        return [shape_a[0], shape_b[0]]

    def call(self, inputs, **kwargs):
        a, b = inputs
        return tf.matmul(a, b, transpose_b=True)


def Bloodhound0(
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
        feature_dim=None,
        model_name=None,
):
    """
    q -> encoder -> -----> zq
                  |
                  v
          c -> decoder -> zc
    """
    enc = backbones.ViTB16(
        weights="imagenet21k+_224",
        include_top=False,
    )
    patch_size = enc.get_layer("patch_embeddings").get_layer("embedding").kernel_size

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

    enc = tf.keras.Model(
        inputs=enc.inputs, outputs=enc.get_layer("encoder").output, name="encoder"
    )
    embed = tf.keras.Model(
        inputs=enc.inputs, outputs=enc.get_layer("pos_embedding").output, name="embed"
    )
    q_enc = enc(inputs_q)
    c_enc = embed(inputs_c)

    x_c = tf.keras.layers.Dropout(dropout_rate)(c_enc)
    x_c = Decoder(
        embed_dim=x_c.shape[-1],
        num_heads=n_heads,
        ff_dim=ff_dim,
        num_layers=n_layers,
        attention_dropout_rate=dropout_rate,
        dense_dropout_rate=dropout_rate,
        norm_output=True,
        pre_norm=False,
        causal=False,
    )([x_c, q_enc])

    x_q = _pool(q_enc, method=pooling, prefix="q_")
    x_c = _pool(x_c, method=pooling, prefix="c_")

    if include_top:
        if pooling is None:
            raise ValueError("`include_top=True` requires `pooling` to be either 'avg', 'max', 'sum', or 'cls'.")

        if feature_dim is not None:
            x_q = tf.keras.layers.Dense(feature_dim)(x_q)
            x_c = tf.keras.layers.Dense(feature_dim)(x_c)

        x_q = L2Normalization(axis=-1)(x_q)
        x_c = L2Normalization(axis=-1)(x_c)
        x = Matmul()([x_q, x_c])
    else:
        x = [x_q, x_c]

    model = tf.keras.Model(
        inputs=[inputs_q, inputs_c], outputs=x, name=model_name
    )
    return model


def Bloodhound1(
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
    feature_dim=None,
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
        include_top=False,
    )
    patch_size = enc.get_layer("patch_embeddings").get_layer("embedding").kernel_size

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

    enc = tf.keras.Model(
        inputs=enc.inputs, outputs=enc.get_layer("encoder").output, name="encoder"
    )
    q_enc = enc(inputs_q)
    c_enc = enc(inputs_c)
    # TODO: positional embedding here too?
    x_c = Decoder(
        embed_dim=c_enc.shape[-1],
        num_heads=n_heads,
        ff_dim=ff_dim,
        num_layers=n_layers,
        attention_dropout_rate=dropout_rate,
        dense_dropout_rate=dropout_rate,
        norm_output=True,
        pre_norm=False,
        causal=False,
    )([c_enc, q_enc])

    x_q = _pool(q_enc, method=pooling, prefix="q_")
    x_c = _pool(x_c, method=pooling, prefix="c_")

    if include_top:
        if pooling is None:
            raise ValueError("`include_top=True` requires `pooling` to be either 'avg', 'max', 'sum', or 'cls'.")

        if feature_dim is not None:
            x_q = tf.keras.layers.Dense(feature_dim)(x_q)
            x_c = tf.keras.layers.Dense(feature_dim)(x_c)

        x_q = L2Normalization(axis=-1)(x_q)
        x_c = L2Normalization(axis=-1)(x_c)
        x = Matmul()([x_q, x_c])
    else:
        x = [x_q, x_c]

    model = tf.keras.Model(
        inputs=[inputs_q, inputs_c], outputs=x, name=model_name
    )
    return model
