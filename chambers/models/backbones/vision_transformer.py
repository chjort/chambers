import tensorflow as tf
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.utils import layer_utils

from chambers.layers.embedding import ConcatEmbedding, LearnedEmbedding1D
from chambers.layers.transformer import Encoder
from chambers.utils.layer_utils import inputs_to_input_layer

BASE_WEIGHTS_PATH = "https://github.com/chjort/chambers/releases/download/v1.0/"
WEIGHTS_HASHES = {
    # imagenet21k (pre-trained on imagenet21k)
    # 'vitb16':
    #     ('ff0ce1ed5accaad05d113ecef2d29149', '043777781b0d5ca756474d60bf115ef1'),
    # 'vitb32':
    #     ('5c31adee48c82a66a32dee3d442f5be8', '1c373b0c196918713da86951d1239007'),
    # 'vitl16':
    #     ('96fc14e3a939d4627b0174a0e80c7371', 'f58d4c1a511c7445ab9a2c2b83ee4e7b'),
    # 'vitl32':
    #     ('5310dcd58ed573aecdab99f8df1121d5', 'b0f23d2e1cd406d67335fb92d85cc279'),
    # imagenet21k+imagenet2012 (pre-trained on imagenet21k and fine-tuned on imagenet2012)
    # 'vitb16':
    #     ('ff0ce1ed5accaad05d113ecef2d29149', '043777781b0d5ca756474d60bf115ef1'),
    # 'vitb32':
    #     ('5c31adee48c82a66a32dee3d442f5be8', '1c373b0c196918713da86951d1239007'),
    # 'vitl16':
    #     ('96fc14e3a939d4627b0174a0e80c7371', 'f58d4c1a511c7445ab9a2c2b83ee4e7b'),
    # 'vitl32':
    #     ('5310dcd58ed573aecdab99f8df1121d5', 'b0f23d2e1cd406d67335fb92d85cc279'),
    # imagenet2012
    # deit
}


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
    weights=None,
    pooling=None,
    feature_dim=None,
    classes=1000,
    classifier_activation=None,
    name=None,
):
    # TODO: validate that feature_dim and weights are mutually exclusive
    # TODO: if weights are given and feature_dim is None, set appropriate feature_dim

    input_shape = imagenet_utils.obtain_input_shape(
        input_shape=input_shape,
        default_size=224,
        min_size=patch_size,
        data_format=tf.keras.backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )
    inputs = inputs_to_input_layer(input_tensor, input_shape)

    if None in inputs.shape[1:]:
        raise ValueError(
            "Input shape must be fully specified; got input shape {}.".format(
                inputs.shape[1:]
            )
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
        initializer="zeros",
        name="add_cls_token",
    )(x)
    x = LearnedEmbedding1D(
        initializer=tf.keras.initializers.RandomNormal(stddev=0.06),
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

    model = tf.keras.models.Model(inputs=inputs, outputs=x, name=name)

    if weights:
        # TODO: load pretrained weights

        pass

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
    weights=None,
    pooling=None,
    classes=1000,
    classifier_activation=None,
    name=None,
):
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape=input_shape,
        default_size=224,
        min_size=patch_size,
        data_format=tf.keras.backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )
    inputs = inputs_to_input_layer(input_tensor, input_shape)

    if None in inputs.shape[1:]:
        raise ValueError(
            "Input shape must be fully specified; got input shape {}.".format(
                inputs.shape[1:]
            )
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
        initializer="zeros",
        name="add_dist_token",
    )(x)
    x = ConcatEmbedding(
        n_embeddings=1,
        embedding_dim=patch_dim,
        side="left",
        axis=1,
        initializer="zeros",
        name="add_cls_token",
    )(x)
    x = LearnedEmbedding1D(
        initializer=tf.keras.initializers.RandomNormal(stddev=0.06),
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

    model = tf.keras.models.Model(inputs=inputs, outputs=x, name=name)

    if weights:
        # TODO: load pretrained weights

        pass

    return model


def ViTB16(
    input_tensor=None,
    input_shape=None,
    include_top=True,
    weights=None,
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
        name="vitb16",
    )
    return vit


def ViTB32(
    input_tensor=None,
    input_shape=None,
    include_top=True,
    weights=None,
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
        name="vitb32",
    )
    return vit


def ViTL16(
    input_tensor=None,
    input_shape=None,
    include_top=True,
    weights=None,
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
        name="vitl16",
    )
    return vit


def ViTL32(
    input_tensor=None,
    input_shape=None,
    include_top=True,
    weights=None,
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
        name="vitl32",
    )
    return vit


def DeiTB16(
    return_dist_token=True,
    input_tensor=None,
    input_shape=None,
    include_top=True,
    weights=None,
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
        name="deitb16",
    )
    return deit
