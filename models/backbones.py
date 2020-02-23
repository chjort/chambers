import os

import tensorflow as tf


def BN_Inception(input_shape=None, freeze_layers=False):
    model_path = os.path.expanduser("~/.keras/models/BN-Inception_notop.h5")
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
    resnet = tf.keras.applications.ResNet50(input_shape=input_shape, weights="imagenet", include_top=include_top)
    if freeze_layers:
        for layer in resnet.layers:
            layer.trainable = False

    return resnet
