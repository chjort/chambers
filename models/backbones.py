import tensorflow as tf


def BN_Inception_ImageNet(input_shape=None, freeze_layers=False):
    model_path = tf.keras.utils.get_file("BN-Inception_notop.h5",
                                         "https://drive.google.com/uc?export=download&id=1eqId67njyNaTe3G2mqjb2fWtGE_5XotY",
                                         cache_subdir='models',
                                         file_hash="7eb8291a8e70fccbccc3bc2fff83311b35d2194ee584c1f1335bb9a240b94145",
                                         hash_algorithm="sha256")
    model = tf.keras.models.load_model(model_path, compile=False)

    if freeze_layers:
        for layer in model.layers:
            layer.trainable = False

    return model


def ResNet50_ImageNet(input_shape=None, freeze_layers=False, include_top=False):
    """
    The ResNet50 model loaded with weights trained from the ImageNet dataset.
    The model is built without the classification layer for ImageNet.

    :param input_shape: 3-D Tuple specifying input shape. This is excluding batch dimension
    :param freeze_layers: If True all parameters of the convolutional layers are untrainable.
    :return: tensorflow.keras.Model
    """
    model = tf.keras.applications.ResNet50(input_shape=input_shape, weights="imagenet", include_top=include_top)
    if freeze_layers:
        for layer in model.layers:
            layer.trainable = False

    return model


def ResNet101_ImageNet(input_shape=None, freeze_layers=False, include_top=False):
    """
    The ResNet50 model loaded with weights trained from the ImageNet dataset.
    The model is built without the classification layer for ImageNet.

    :param input_shape: 3-D Tuple specifying input shape. This is excluding batch dimension
    :param freeze_layers: If True all parameters of the convolutional layers are untrainable.
    :return: tensorflow.keras.Model
    """
    model = tf.keras.applications.ResNet101(input_shape=input_shape, weights="imagenet", include_top=include_top)
    if freeze_layers:
        for layer in model.layers:
            layer.trainable = False

    return model


def ResNext50_ImageNet(input_shape=None, freeze_layers=False, include_top=False):
    model = keras_applications.resnext.ResNeXt50(input_shape=input_shape, weights="imagenet", include_top=include_top,
                                                 backend=tf.keras.backend,
                                                 layers=tf.keras.layers,
                                                 models=tf.keras.models,
                                                 utils=tf.keras.utils)
    if freeze_layers:
        for layer in model.layers:
            layer.trainable = False

    return model


def ResNext101_ImageNet(input_shape=None, freeze_layers=False, include_top=False):
    model = keras_applications.resnext.ResNeXt101(input_shape=input_shape, weights="imagenet", include_top=include_top,
                                                  backend=tf.keras.backend,
                                                  layers=tf.keras.layers,
                                                  models=tf.keras.models,
                                                  utils=tf.keras.utils
                                                  )
    if freeze_layers:
        for layer in model.layers:
            layer.trainable = False

    return model
