import tensorflow as tf
from tensorflow.python.keras.utils import data_utils

from chambers.augmentations import ImageNetNormalization

BASE_WEIGHTS_PATH = (
    'https://github.com/chjort/chambers/releases/download/v1.0/')
WEIGHTS_HASHES = {
    'bninception':
        (None, '7eb8291a8e70fccbccc3bc2fff83311b35d2194ee584c1f1335bb9a240b94145')
}


def BNInception(
    input_tensor=None,
    pooling=None,
):

    model_name = "bninception"
    file_name = model_name + '_imagenet_1000_no_top.h5'
    file_hash = WEIGHTS_HASHES[model_name][1]
    model_path = data_utils.get_file(
        file_name,
        BASE_WEIGHTS_PATH + file_name,
        cache_subdir='models',
        file_hash=file_hash,
        hash_algorithm="sha256"
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

    return tf.keras.Model(inputs=img_input, outputs=x, name=model_name)


preprocess_input = ImageNetNormalization(mode="tf", name="inception_preprocess")
