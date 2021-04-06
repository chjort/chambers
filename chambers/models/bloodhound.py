import tensorflow as tf

from chambers.layers.reduce import Sum
from chambers.layers.transformer import Decoder
from chambers.models import backbones
from chambers.models.backbones.vision_transformer import _obtain_inputs
from chambers.layers.normalization import L2Normalization


def Bloodhound0(
    n_layers,
    n_heads,
    ff_dim,
    dropout_rate=0.1,
    input_tensor1=None,
    input_shape1=None,
    input_tensor2=None,
    input_shape2=None,
    include_top=True,
    weights="imagenet21k+_224",
    pooling=None,
    feature_dim=None,
    model_name=None,
):
    """
    c -> encoder ->
                  |
                  v
          q -> decoder -> z (1d)

    """
    enc = backbones.ViTB16(
        weights="imagenet21k+_224",
        include_top=False,
    )
    patch_size = enc.get_layer("patch_embeddings").get_layer("embedding").kernel_size

    inputs1 = _obtain_inputs(
        input_tensor1,
        input_shape1,
        default_size=224,
        min_size=patch_size,
        weights=weights,
        model_name=model_name,
        name="x1",
    )
    inputs2 = _obtain_inputs(
        input_tensor2,
        input_shape2,
        default_size=224,
        min_size=patch_size,
        weights=weights,
        model_name=model_name,
        name="x2",
    )

    enc = tf.keras.Model(
        inputs=enc.inputs, outputs=enc.get_layer("encoder").output, name="encoder"
    )
    x_enc = enc(inputs1)

    embed = tf.keras.Model(
        inputs=enc.inputs, outputs=enc.get_layer("pos_embedding").output, name="embed"
    )
    x = embed(inputs2)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = Decoder(
        embed_dim=x.shape[-1],
        num_heads=n_heads,
        ff_dim=ff_dim,
        num_layers=n_layers,
        attention_dropout_rate=dropout_rate,
        dense_dropout_rate=dropout_rate,
        norm_output=True,
        pre_norm=False,
        causal=False,
    )([x, x_enc])

    if pooling == "avg":
        x = tf.keras.layers.Cropping1D((1, 0), name="sequence_embeddings")(x)
        x = tf.keras.layers.GlobalAveragePooling1D(name="avg_pool")(x)
    elif pooling == "max":
        x = tf.keras.layers.Cropping1D((1, 0), name="sequence_embeddings")(x)
        x = tf.keras.layers.GlobalMaxPooling1D(name="max_pool")(x)
    elif pooling == "cls":
        x = tf.keras.Sequential(
            [
                tf.keras.layers.Cropping1D((0, x.shape[1] - 1)),
                tf.keras.layers.Reshape([-1]),
            ],
            name="cls_embedding",
        )(x)
    else:
        x = tf.keras.layers.Cropping1D((1, 0), name="sequence_embeddings")(x)
        x = Sum(axis=1, name="sum_pool")(x)

    if include_top:
        if feature_dim is None:
            feature_dim = x.shape[-1]

        x = tf.keras.layers.Dense(feature_dim)(x)
        x = L2Normalization(axis=-1)(x)

    model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=x)
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
        c -> encoder ->
                      |
                      v
    q -> encoder -> decoder -> z (1d)

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
    x = Decoder(
        embed_dim=q_enc.shape[-1],
        num_heads=n_heads,
        ff_dim=ff_dim,
        num_layers=n_layers,
        attention_dropout_rate=dropout_rate,
        dense_dropout_rate=dropout_rate,
        norm_output=True,
        pre_norm=False,
        causal=False,
    )([q_enc, c_enc])

    if pooling == "avg":
        x = tf.keras.layers.Cropping1D((1, 0), name="sequence_embeddings")(x)
        x = tf.keras.layers.GlobalAveragePooling1D(name="avg_pool")(x)
    elif pooling == "max":
        x = tf.keras.layers.Cropping1D((1, 0), name="sequence_embeddings")(x)
        x = tf.keras.layers.GlobalMaxPooling1D(name="max_pool")(x)
    elif pooling == "cls":
        x = tf.keras.Sequential(
            [
                tf.keras.layers.Cropping1D((0, x.shape[1] - 1)),
                tf.keras.layers.Reshape([-1]),
            ],
            name="cls_embedding",
        )(x)
    else:
        x = tf.keras.layers.Cropping1D((1, 0), name="sequence_embeddings")(x)
        x = Sum(axis=1, name="sum_pool")(x)

    if include_top:
        if feature_dim is None:
            feature_dim = x.shape[-1]

        x = tf.keras.layers.Dense(feature_dim)(x)
        # x = L2Normalization(axis=-1)(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid", dtype=tf.float32)(x)

    model = tf.keras.Model(inputs=[inputs_q, inputs_c], outputs=x)
    return model
