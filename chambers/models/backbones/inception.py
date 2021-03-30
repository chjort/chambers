import tensorflow as tf
from tensorflow.python.lib.io import file_io

from chambers.augmentations import ImageNetNormalization


def BNInception(
    weights="imagenet",
    input_tensor=None,
    pooling=None,
):
    if not (weights in {"imagenet", None} or file_io.file_exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded."
        )

    model_path = tf.keras.utils.get_file(
        "BN-Inception_notop.h5",
        "https://drive.google.com/uc?export=download&id=1eqId67njyNaTe3G2mqjb2fWtGE_5XotY",
        cache_subdir="models",
        file_hash="7eb8291a8e70fccbccc3bc2fff83311b35d2194ee584c1f1335bb9a240b94145",
        hash_algorithm="sha256",
    )

    model = tf.keras.models.load_model(model_path, compile=False)

    if input_tensor is None:
        img_input = model.input
        x = model.output
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor=input_tensor, shape=None)
        else:
            img_input = input_tensor
        x = model(img_input)

    if pooling == "avg":
        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    elif pooling == "max":
        x = tf.keras.layers.GlobalMaxPooling2D(name="max_pool")(x)

    return tf.keras.Model(inputs=img_input, outputs=x, name="BNInception")


preprocess_input = ImageNetNormalization(mode="tf", name="inception_preprocess")
