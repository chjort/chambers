from tensorflow.keras.layers import Flatten, ReLU, Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense
from tensorflow.keras.models import Model

from .backbones import ResNet50_ImageNet, BN_Inception
from ..layers import ConvBlock, GlobalGeneralizedMean, L2Normalization


def BN_Inception_SPoC(input_shape=None, freeze_layers=False):
    bn_inception = BN_Inception(input_shape, freeze_layers)
    x = GlobalAveragePooling2D()(bn_inception.output)
    x = Dense(units=512)(x)
    x = L2Normalization(axis=1)(x)

    model = Model(inputs=bn_inception.input, outputs=x, name="BN_Inception_DenseL2")
    return model


def ResNet50_MAC(input_shape=None, freeze_layers=False):
    resnet = ResNet50_ImageNet(input_shape, freeze_layers)

    # x = ReLU()(resnet.output)
    x = GlobalMaxPooling2D()(resnet.output)
    x = Dense(512)(x)
    x = L2Normalization(axis=1)(x)

    model = Model(inputs=resnet.input, outputs=x, name="ResNet50_MAC")
    return model


def ResNet50_SPoC(input_shape=None, freeze_layers=False):
    resnet = ResNet50_ImageNet(input_shape, freeze_layers)

    # x = ReLU()(resnet.output)
    x = GlobalAveragePooling2D()(resnet.output)
    x = Dense(512)(x)
    x = L2Normalization(axis=1)(x)

    model = Model(inputs=resnet.input, outputs=x, name="ResNet50_SPoC")
    return model


def ResNet50_GeM(input_shape=None, freeze_layers=False):
    """
    The ResNet50 with Global Generalized Mean pooling (GeM) as the last layer. The model is loaded with
    weights trained from the ImageNet dataset. The model is built without the classification layer for ImageNet.

    :param input_shape: 3-D Tuple specifying input shape. This is excluding batch dimension
    :param freeze_layers: If True all parameters of the convolutional layers are untrainable.
    :return: tensorflow.keras.Model
    """
    resnet = ResNet50_ImageNet(input_shape, freeze_layers)

    # x = ReLU()(resnet.output)
    x = GlobalGeneralizedMean(p=3, trainable=False)(resnet.output)
    x = Dense(512)(x)
    x = L2Normalization(axis=1)(x)

    model = Model(inputs=resnet.input, outputs=x, name="ResNet50_GeM")

    return model


def CNN(input_shape=None):
    """
    A Convolutional Neural Network comprised of four convolutional blocks.
    In each block an input is passed through Conv2D, Batch Normalization, ReLU activation, and Max-pooling

    """
    if input_shape is None:
        input_shape = (None, None, 3)

    input_ = Input(shape=input_shape)
    x = ConvBlock(filters=64, conv_size=(3, 3))(input_)
    x = ConvBlock(filters=64, conv_size=(3, 3))(x)
    x = ConvBlock(filters=64, conv_size=(3, 3))(x)
    x = ConvBlock(filters=64, conv_size=(3, 3))(x)
    x = Flatten()(x)
    model = Model(inputs=input_, outputs=x, name="cnn")

    return model
