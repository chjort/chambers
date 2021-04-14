import tensorflow as tf
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils

from chambers.augmentations import ImageNetNormalization
from chambers.layers.embedding import ConcatEmbedding, LearnedEmbedding1D
from chambers.layers.transformer import Encoder
from chambers.utils.layer_utils import inputs_to_input_layer

# imagenet21k (pre-trained on imagenet21k)
# imagenet21k+ (pre-trained on imagenet21k and fine-tuned on imagenet2012)
# imagenet (pre-trained on imagenet2012)
BASE_WEIGHTS_PATH = "https://github.com/chjort/chambers/releases/download/v1.1/"
WEIGHTS_HASHES = {
    # model_name: {weight: (top_hash, no_top_hash, suffix)}
    "vits16": {
        "imagenet_224_deit": (
            "6df5bc5734ace3fc83e4a2e826cfe37c",
            "3ddca7413a039e9a8979c1718e33c597",
            "imagenet_1000_224_deit",
        ),
    },
    "vitb16": {
        "imagenet21k": (None, "7600a249df4c5460e16ee8637a104683", "imagenet_21k_224"),
        "imagenet21k+_224": (
            "6c987252c94ae15c34e4b2ef8b69b026",
            "fb29e40486b4dd1b82ac8635555bed65",
            "imagenet_21k_1000_224",
        ),
        "imagenet21k+_384": (
            "f189719ecc305d0ccd9525206f741409",
            "e69336a399b1a334adf72ad237df2c30",
            "imagenet_21k_1000_384",
        ),
        "imagenet_224_deit": (
            "b313ff9ff936ac4639199e8c28cf2ca4",
            "600c2033dc9f53181147596c867f62f6",
            "imagenet_21k_1000_224_deit",
        ),
        "imagenet_384_deit": (
            "134ee39f1a10c276f528b521a4353647",
            "e3a4c07722b7e3a62cbf4b2c137759e3",
            "imagenet_21k_1000_384_deit",
        ),
    },
    "vitb32": {
        "imagenet21k": (None, "14f8c10584cf61786a658723cc8d1b68", "imagenet_21k_224"),
        "imagenet21k+_384": (
            "d4b41bf765992566151f5915cc1b275b",
            "aa8863a833d9e3e592768c5c95d74361",
            "imagenet_21k_1000_384",
        ),
    },
    "vitl16": {
        "imagenet21k": (None, "ad70eb7a7a50daf3c96a790b2f7c38ca", "imagenet_21k_224"),
        "imagenet21k+_224": (
            "c39ee61dfd071a1e1a8994fed58dec35",
            "51dbbcabe79feb81237369909dc14d2e",
            "imagenet_21k_1000_224",
        ),
        "imagenet21k+_384": (
            "451f946387516c835f576dff7b5074f5",
            "a0775f7493bd816fcb0513fb813d180c",
            "imagenet_21k_1000_384",
        ),
    },
    "vitl32": {
        "imagenet21k": (None, "645d669250d87f5d8ba0a2fb1188c510", "imagenet_21k_224"),
        "imagenet21k+_384": (
            "8aacec1f38deaec287b2122ded1bbff4",
            "6aa0e4197259e0a369972221af546cf0",
            "imagenet_21k_1000_384",
        ),
    },
    "deits16": {
        "imagenet_224": (
            "309350442160f3e9bc325a0cdeac49ef",
            "bf207ba3aeb8ec578eb0c5157192f59c",
            "imagenet_1000_224",
        ),
    },
    "deitb16": {
        "imagenet_224": (
            "898b74940e3a61e90b802dae47af4428",
            "2ae45d564218b76fea4aa03cc0db279b",
            "imagenet_1000_224",
        ),
        "imagenet_384": (
            "ca3e7ca40e4b96ead9508ea1e5e35893",
            "1e3be99ad5acc90101f80e94469c815e",
            "imagenet_1000_384",
        ),
    },
}


def _are_weights_pretrained(weights, model_name):
    return (model_name in WEIGHTS_HASHES) and (weights in WEIGHTS_HASHES[model_name])


def _get_model_info(weights, model_name):
    if _are_weights_pretrained(weights, model_name):
        weight_info = WEIGHTS_HASHES[model_name][weights]
        weight_suffix = weight_info[2]
        weight_suffix = weight_suffix.replace("_deit", "")
        default_size = int(weight_suffix.split("_")[-1])
        has_feature = "21k" in weight_suffix and "1000" not in weight_suffix
    else:
        default_size = 224
        has_feature = False

    return default_size, has_feature


def _obtain_inputs(
    input_tensor, input_shape, default_size, min_size, weights, model_name, name=None
):
    if input_shape is not None and _are_weights_pretrained(weights, model_name):
        default_shape = (default_size, default_size, input_shape[-1])
        if tuple(input_shape) != default_shape:
            raise ValueError(
                "Weights '{}' require `input_shape` to be {}.".format(
                    weights, default_shape
                )
            )

    # TODO: write own `obtain_input_shape` function
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape=input_shape,
        default_size=default_size,
        min_size=min_size,
        data_format=tf.keras.backend.image_data_format(),
        require_flatten=(input_shape is None),
        weights="imagenet" if weights else None,
    )
    inputs = inputs_to_input_layer(input_tensor, input_shape, name)

    if None in inputs.shape[1:]:
        raise ValueError(
            "Input shape must be fully specified; got input shape {}.".format(
                inputs.shape[1:]
            )
        )
    return inputs


def _load_weights(model, weights, include_top):
    model_name = model.name

    if _are_weights_pretrained(weights, model_name):
        weight_info = WEIGHTS_HASHES[model_name][weights]
        file_suffix = weight_info[2]
        if include_top:
            file_name = model_name + "_" + file_suffix + ".h5"
            file_hash = weight_info[0]
        else:
            file_name = model_name + "_" + file_suffix + "_no_top.h5"
            file_hash = weight_info[1]
        weights_path = data_utils.get_file(
            file_name,
            BASE_WEIGHTS_PATH + file_name,
            cache_subdir="models",
            file_hash=file_hash,
        )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)


def VisionTransformer(
    patch_size,
    patch_dim,
    n_encoder_layers,
    n_heads,
    ff_dim,
    dropout_rate=0.1,
    input_tensor=None,
    input_shape=None,
    include_top=True,
    weights="imagenet21k+_224",
    pooling=None,
    feature_dim=None,
    classes=1000,
    classifier_activation=None,
    model_name=None,
):
    weights_are_pretrained = _are_weights_pretrained(weights, model_name)
    default_size, has_feature = _get_model_info(weights, model_name)

    if weights_are_pretrained and feature_dim is not None:
        raise ValueError("'weights' and 'feature_dim' are mutually exclusive.")
    elif weights_are_pretrained and has_feature:
        feature_dim = patch_dim
        if include_top:
            print(
                "Warning: weights '{}' has no top. 'include_top' will be set to False.".format(
                    weights
                )
            )
            include_top = False

    inputs = _obtain_inputs(
        input_tensor,
        input_shape,
        default_size=default_size,
        min_size=patch_size,
        weights=weights,
        model_name=model_name,
    )

    patch_embeddings = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=patch_dim,
                kernel_size=patch_size,
                strides=patch_size,
                padding="valid",
                name="embedding",
            ),
            tf.keras.layers.Reshape([-1, patch_dim]),
        ],
        name="patch_embeddings",
    )
    x = patch_embeddings(inputs)
    x = ConcatEmbedding(
        n_embeddings=1,
        embedding_dim=patch_dim,
        side="left",
        axis=1,
        initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        name="add_cls_token",
    )(x)
    x = LearnedEmbedding1D(
        initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        name="pos_embedding",
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = Encoder(
        embed_dim=patch_dim,
        num_heads=n_heads,
        ff_dim=ff_dim,
        num_layers=n_encoder_layers,
        attention_dropout_rate=dropout_rate,
        dense_dropout_rate=dropout_rate,
        pre_norm=True,
        norm_output=True,
    )(x)

    if pooling == "avg":
        x = tf.keras.layers.Cropping1D((1, 0), name="sequence_embeddings")(x)
        x = tf.keras.layers.GlobalAveragePooling1D(name="avg_pool")(x)
    elif pooling == "max":
        x = tf.keras.layers.Cropping1D((1, 0), name="sequence_embeddings")(x)
        x = tf.keras.layers.GlobalMaxPooling1D(name="max_pool")(x)
    else:
        x = tf.keras.Sequential(
            [
                tf.keras.layers.Cropping1D((0, x.shape[1] - 1)),
                tf.keras.layers.Reshape([-1]),
            ],
            name="cls_embedding",
        )(x)

    if feature_dim is not None:
        x = tf.keras.layers.Dense(units=feature_dim, activation="tanh", name="feature")(
            x
        )

    if include_top:
        x = tf.keras.layers.Dense(
            units=classes, activation=classifier_activation, name="predictions"
        )(x)

    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)

    model = tf.keras.models.Model(inputs=inputs, outputs=x, name=model_name)

    _load_weights(model, weights, include_top)

    return model


def DistilledVisionTransformer(
    patch_size,
    patch_dim,
    n_encoder_layers,
    n_heads,
    ff_dim,
    dropout_rate=0.1,
    return_dist_token=True,
    input_tensor=None,
    input_shape=None,
    include_top=True,
    weights="imagenet_224",
    pooling=None,
    classes=1000,
    classifier_activation=None,
    model_name=None,
):
    default_size, has_feature = _get_model_info(weights, model_name)

    inputs = _obtain_inputs(
        input_tensor,
        input_shape,
        default_size=default_size,
        min_size=patch_size,
        weights=weights,
        model_name=model_name,
    )

    patch_embeddings = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=patch_dim,
                kernel_size=patch_size,
                strides=patch_size,
                padding="valid",
                name="embedding",
            ),
            tf.keras.layers.Reshape([-1, patch_dim]),
        ],
        name="patch_embeddings",
    )
    x = patch_embeddings(inputs)
    x = ConcatEmbedding(
        n_embeddings=1,
        embedding_dim=patch_dim,
        side="left",
        axis=1,
        initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        name="add_dist_token",
    )(x)
    x = ConcatEmbedding(
        n_embeddings=1,
        embedding_dim=patch_dim,
        side="left",
        axis=1,
        initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        name="add_cls_token",
    )(x)
    x = LearnedEmbedding1D(
        initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        name="pos_embedding",
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = Encoder(
        embed_dim=patch_dim,
        num_heads=n_heads,
        ff_dim=ff_dim,
        num_layers=n_encoder_layers,
        attention_dropout_rate=dropout_rate,
        dense_dropout_rate=dropout_rate,
        pre_norm=True,
        norm_output=True,
    )(x)

    if pooling == "avg":
        x_cls = tf.keras.layers.Cropping1D((2, 0), name="sequence_embeddings")(x)
        x_cls = tf.keras.layers.GlobalAveragePooling1D(name="avg_pool")(x_cls)
    elif pooling == "max":
        x_cls = tf.keras.layers.Cropping1D((2, 0), name="sequence_embeddings")(x)
        x_cls = tf.keras.layers.GlobalMaxPooling1D(name="max_pool")(x_cls)
    else:
        x_cls = tf.keras.Sequential(
            [
                tf.keras.layers.Cropping1D((0, x.shape[1] - 1)),
                tf.keras.layers.Reshape([-1]),
            ],
            name="cls_embedding",
        )(x)

    x_dist = tf.keras.Sequential(
        [
            tf.keras.layers.Cropping1D((1, x.shape[1] - 2)),
            tf.keras.layers.Reshape([-1]),
        ],
        name="dist_embedding",
    )(x)

    if include_top:
        x_cls = tf.keras.layers.Dense(
            units=classes, activation=classifier_activation, name="predictions"
        )(x_cls)
        x_dist = tf.keras.layers.Dense(
            units=classes, activation=classifier_activation, name="predictions_dist"
        )(x_dist)

    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)

    if return_dist_token:
        x = [x_cls, x_dist]
    else:
        x = tf.keras.layers.Average()([x_cls, x_dist])

    x = tf.keras.layers.Activation("linear", dtype=tf.float32, name="cast_float32")(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x, name=model_name)

    _load_weights(model, weights, include_top)

    return model


def ViTS16(
    input_tensor=None,
    input_shape=None,
    include_top=True,
    weights="imagenet_224_deit",
    pooling=None,
    feature_dim=None,
    classes=1000,
    classifier_activation=None,
):
    patch_size = 16
    patch_dim = 384
    n_encoder_layers = 12
    n_heads = 6
    ff_dim = 1536

    vit = VisionTransformer(
        patch_size=patch_size,
        patch_dim=patch_dim,
        n_encoder_layers=n_encoder_layers,
        n_heads=n_heads,
        ff_dim=ff_dim,
        dropout_rate=0.1,
        feature_dim=feature_dim,
        input_tensor=input_tensor,
        input_shape=input_shape,
        include_top=include_top,
        weights=weights,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        model_name="vits16",
    )
    return vit


def ViTB16(
    input_tensor=None,
    input_shape=None,
    include_top=True,
    weights="imagenet21k+_224",
    pooling=None,
    feature_dim=None,
    classes=1000,
    classifier_activation=None,
):
    patch_size = 16
    patch_dim = 768
    n_encoder_layers = 12
    n_heads = 12
    ff_dim = 3072

    vit = VisionTransformer(
        patch_size=patch_size,
        patch_dim=patch_dim,
        n_encoder_layers=n_encoder_layers,
        n_heads=n_heads,
        ff_dim=ff_dim,
        dropout_rate=0.1,
        feature_dim=feature_dim,
        input_tensor=input_tensor,
        input_shape=input_shape,
        include_top=include_top,
        weights=weights,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        model_name="vitb16",
    )
    return vit


def ViTB32(
    input_tensor=None,
    input_shape=None,
    include_top=True,
    weights="imagenet21k+_384",
    pooling=None,
    feature_dim=None,
    classes=1000,
    classifier_activation=None,
):
    patch_size = 32
    patch_dim = 768
    n_encoder_layers = 12
    n_heads = 12
    ff_dim = 3072

    vit = VisionTransformer(
        patch_size=patch_size,
        patch_dim=patch_dim,
        n_encoder_layers=n_encoder_layers,
        n_heads=n_heads,
        ff_dim=ff_dim,
        dropout_rate=0.1,
        feature_dim=feature_dim,
        input_tensor=input_tensor,
        input_shape=input_shape,
        include_top=include_top,
        weights=weights,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        model_name="vitb32",
    )
    return vit


def ViTL16(
    input_tensor=None,
    input_shape=None,
    include_top=True,
    weights="imagenet21k+_224",
    pooling=None,
    feature_dim=None,
    classes=1000,
    classifier_activation=None,
):
    patch_size = 16
    patch_dim = 1024
    n_encoder_layers = 24
    n_heads = 16
    ff_dim = 4096

    vit = VisionTransformer(
        patch_size=patch_size,
        patch_dim=patch_dim,
        n_encoder_layers=n_encoder_layers,
        n_heads=n_heads,
        ff_dim=ff_dim,
        dropout_rate=0.1,
        feature_dim=feature_dim,
        input_tensor=input_tensor,
        input_shape=input_shape,
        include_top=include_top,
        weights=weights,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        model_name="vitl16",
    )
    return vit


def ViTL32(
    input_tensor=None,
    input_shape=None,
    include_top=True,
    weights="imagenet21k+_384",
    pooling=None,
    feature_dim=None,
    classes=1000,
    classifier_activation=None,
):
    patch_size = 32
    patch_dim = 1024
    n_encoder_layers = 24
    n_heads = 16
    ff_dim = 4096

    vit = VisionTransformer(
        patch_size=patch_size,
        patch_dim=patch_dim,
        n_encoder_layers=n_encoder_layers,
        n_heads=n_heads,
        ff_dim=ff_dim,
        dropout_rate=0.1,
        feature_dim=feature_dim,
        input_tensor=input_tensor,
        input_shape=input_shape,
        include_top=include_top,
        weights=weights,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        model_name="vitl32",
    )
    return vit


def DeiTS16(
    return_dist_token=True,
    input_tensor=None,
    input_shape=None,
    include_top=True,
    weights="imagenet_224",
    pooling=None,
    classes=1000,
    classifier_activation=None,
):
    patch_size = 16
    patch_dim = 384
    n_encoder_layers = 12
    n_heads = 6
    ff_dim = 1536

    deit = DistilledVisionTransformer(
        patch_size=patch_size,
        patch_dim=patch_dim,
        n_encoder_layers=n_encoder_layers,
        n_heads=n_heads,
        ff_dim=ff_dim,
        dropout_rate=0.1,
        return_dist_token=return_dist_token,
        input_tensor=input_tensor,
        input_shape=input_shape,
        include_top=include_top,
        weights=weights,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        model_name="deits16",
    )
    return deit


def DeiTB16(
    return_dist_token=True,
    input_tensor=None,
    input_shape=None,
    include_top=True,
    weights="imagenet_224",
    pooling=None,
    classes=1000,
    classifier_activation=None,
):
    patch_size = 16
    patch_dim = 768
    n_encoder_layers = 12
    n_heads = 12
    ff_dim = 3072

    deit = DistilledVisionTransformer(
        patch_size=patch_size,
        patch_dim=patch_dim,
        n_encoder_layers=n_encoder_layers,
        n_heads=n_heads,
        ff_dim=ff_dim,
        dropout_rate=0.1,
        return_dist_token=return_dist_token,
        input_tensor=input_tensor,
        input_shape=input_shape,
        include_top=include_top,
        weights=weights,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        model_name="deitb16",
    )
    return deit


preprocess_input = ImageNetNormalization(mode="tf", name="vit_preprocess")
