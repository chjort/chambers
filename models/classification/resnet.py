"""
keras_resnet.models._2d
~~~~~~~~~~~~~~~~~~~~~~~

This module implements popular two-dimensional residual models.
"""

import tensorflow as tf
from tensorflow.python.keras import layers


resnet_filename = 'ResNet-{}-model.keras.h5'
resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)


def download_resnet_imagenet(v):
    v = int(v.replace('resnet', ''))

    filename = resnet_filename.format(v)
    resource = resnet_resource.format(v)
    if v == 50:
        checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
    elif v == 101:
        checksum = '05dc86924389e5b401a9ea0348a3213c'
    elif v == 152:
        checksum = '6ee11ef2b135592f8031058820bb9e71'
    else:
        return ValueError("Invalid ResNet version")

    return tf.keras.utils.get_file(
        filename,
        resource,
        cache_subdir='models',
        md5_hash=checksum
)


def ResNet(input_tensor, blocks, block, include_top=True, classes=1000, numerical_names=None, name="ResNet", *args, **kwargs):
    """
    Constructs a `tf.keras.models.Model` object using the given block count.

    :param input_tensor: input tensor (e.g. an instance of `layers.Input`)

    :param blocks: the network’s residual architecture

    :param block: a residual block (e.g. an instance of `keras_resnet.blocks.basic_2d`)

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)


    :param numerical_names: list of bool, same size as blocks, used to indicate whether names of layers should include numbers or letters

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >> import keras_resnet.blocks
        >> import keras_resnet.models

        >> shape, classes = (224, 224, 3), 1000

        >> x = layers.Input(shape)

        >> blocks = [2, 2, 2, 2]

        >> block = keras_resnet.blocks.basic_2d

        >> model = keras_resnet.models.ResNet(x, classes, blocks, block, classes=classes)

        >> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if tf.keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if numerical_names is None:
        numerical_names = [True] * len(blocks)

    x = layers.ZeroPadding2D(padding=3, name="padding_conv1")(input_tensor)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name="conv1")(x)
    x = layers.BatchNormalization(axis=axis, epsilon=1e-5, name="bn_conv1")(x)
    x = layers.Activation("relu", name="conv1_relu")(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)

    features = 64

    outputs = []

    for stage_id, iterations in enumerate(blocks):
        for block_id in range(iterations):
            x = block(features, stage_id, block_id, numerical_name=(block_id > 0 and numerical_names[stage_id]))(x)

        features *= 2

        outputs.append(x)

    inputs = tf.keras.utils.get_source_inputs(input_tensor)
    if include_top:
        assert classes > 0

        x = layers.GlobalAveragePooling2D(name="pool5")(x)
        x = layers.Dense(classes, activation="softmax", name="fc1000")(x)

        return tf.keras.models.Model(inputs=inputs, outputs=x, *args, **kwargs)
    else:
        # Else output each stages features
        return tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name, *args, **kwargs)


def ResNet18(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `tf.keras.models.Model` according to the ResNet18 specifications.

    :param inputs: input tensor (e.g. an instance of `layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >> import keras_resnet.models

        >> shape, classes = (224, 224, 3), 1000

        >> x = layers.Input(shape)

        >> model = keras_resnet.models.ResNet18(x, classes=classes)

        >> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [2, 2, 2, 2]

    return ResNet(inputs, blocks, block=basic_2d, include_top=include_top, classes=classes, name="ResNet18", *args, **kwargs)


def ResNet34(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `tf.keras.models.Model` according to the ResNet34 specifications.

    :param inputs: input tensor (e.g. an instance of `layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >> import keras_resnet.models

        >> shape, classes = (224, 224, 3), 1000

        >> x = layers.Input(shape)

        >> model = keras_resnet.models.ResNet34(x, classes=classes)

        >> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [3, 4, 6, 3]

    return ResNet(inputs, blocks, block=basic_2d, include_top=include_top, classes=classes, name="ResNet34", *args, **kwargs)


def ResNet50(inputs, weights="imagenet", blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `tf.keras.models.Model` according to the ResNet50 specifications.

    :param inputs: input tensor (e.g. an instance of `layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >> import keras_resnet.models

        >> shape, classes = (224, 224, 3), 1000

        >> x = layers.Input(shape)

        >> model = keras_resnet.models.ResNet50(x)

        >> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [3, 4, 6, 3]
    numerical_names = [False, False, False, False]

    model = ResNet(inputs, blocks, numerical_names=numerical_names, block=bottleneck_2d, include_top=include_top,
                  classes=classes, name="ResNet50", *args, **kwargs)

    if weights == "imagenet":
        imagenet_weights = download_resnet_imagenet("resnet50")
        model.load_weights(imagenet_weights)
    elif weights is not None:
        model.load_weights(weights)

    return model


def ResNet101(inputs, weights="imagenet", blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `tf.keras.models.Model` according to the ResNet101 specifications.

    :param inputs: input tensor (e.g. an instance of `layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >> import keras_resnet.models

        >> shape, classes = (224, 224, 3), 1000

        >> x = layers.Input(shape)

        >> model = keras_resnet.models.ResNet101(x, classes=classes)

        >> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [3, 4, 23, 3]
    numerical_names = [False, True, True, False]

    model = ResNet(inputs, blocks, numerical_names=numerical_names, block=bottleneck_2d, include_top=include_top,
                  classes=classes, name="ResNet101", *args, **kwargs)

    if weights == "imagenet":
        imagenet_weights = download_resnet_imagenet("resnet101")
        model.load_weights(imagenet_weights)
    elif weights is not None:
        model.load_weights(weights)

    return model


def ResNet152(inputs, weights="imagenet", blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `tf.keras.models.Model` according to the ResNet152 specifications.

    :param inputs: input tensor (e.g. an instance of `layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >> import keras_resnet.models

        >> shape, classes = (224, 224, 3), 1000

        >> x = layers.Input(shape)

        >> model = keras_resnet.models.ResNet152(x, classes=classes)

        >> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [3, 8, 36, 3]
    numerical_names = [False, True, True, False]

    model = ResNet(inputs, blocks, numerical_names=numerical_names, block=bottleneck_2d, include_top=include_top,
                  classes=classes, name="ResNet152", *args, **kwargs)

    if weights == "imagenet":
        imagenet_weights = download_resnet_imagenet("resnet152")
        model.load_weights(imagenet_weights)
    elif weights is not None:
        model.load_weights(weights)

    return model


def ResNet200(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `tf.keras.models.Model` according to the ResNet200 specifications.

    :param inputs: input tensor (e.g. an instance of `layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >> import keras_resnet.models

        >> shape, classes = (224, 224, 3), 1000

        >> x = layers.Input(shape)

        >> model = keras_resnet.models.ResNet200(x, classes=classes)

        >> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [3, 24, 36, 3]
    numerical_names = [False, True, True, False]

    return ResNet(inputs, blocks, numerical_names=numerical_names, block=bottleneck_2d, include_top=include_top,
                  classes=classes, name="ResNet200", *args, **kwargs)


parameters = {
    "kernel_initializer": "he_normal"
}


def basic_2d(filters, stage=0, block=0, kernel_size=3, numerical_name=False, stride=None):
    """
    A two-dimensional basic block.

    :param filters: the output’s feature space

    :param stage: int representing the stage of this block (starting from 0)

    :param block: int representing this block (starting from 0)

    :param kernel_size: size of the kernel

    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})

    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id

    Usage:

        >> import keras_resnet.blocks

        >> keras_resnet.blocks.basic_2d(64)
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if tf.keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = layers.ZeroPadding2D(padding=1, name="padding{}{}_branch2a".format(stage_char, block_char))(x)
        y = layers.Conv2D(filters, kernel_size, strides=stride, use_bias=False,
                                   name="res{}{}_branch2a".format(stage_char, block_char), **parameters)(y)
        y = layers.BatchNormalization(axis=axis, epsilon=1e-5,
                                               name="bn{}{}_branch2a".format(stage_char, block_char))(y)
        y = layers.Activation("relu", name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

        y = layers.ZeroPadding2D(padding=1, name="padding{}{}_branch2b".format(stage_char, block_char))(y)
        y = layers.Conv2D(filters, kernel_size, use_bias=False,
                                   name="res{}{}_branch2b".format(stage_char, block_char), **parameters)(y)
        y = layers.BatchNormalization(axis=axis, epsilon=1e-5,
                                               name="bn{}{}_branch2b".format(stage_char, block_char))(y)

        if block == 0:
            shortcut = layers.Conv2D(filters, (1, 1), strides=stride, use_bias=False,
                                              name="res{}{}_branch1".format(stage_char, block_char), **parameters)(x)
            shortcut = layers.BatchNormalization(axis=axis, epsilon=1e-5,
                                                          name="bn{}{}_branch1".format(stage_char, block_char))(
                shortcut)
        else:
            shortcut = x

        y = layers.Add(name="res{}{}".format(stage_char, block_char))([y, shortcut])
        y = layers.Activation("relu", name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f


def bottleneck_2d(filters, stage=0, block=0, kernel_size=3, numerical_name=False, stride=None):
    """
    A two-dimensional bottleneck block.

    :param filters: the output’s feature space

    :param stage: int representing the stage of this block (starting from 0)

    :param block: int representing this block (starting from 0)

    :param kernel_size: size of the kernel

    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})

    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id

    Usage:

        >> import keras_resnet.blocks

        >> bottleneck_2d(64)
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if tf.keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = layers.Conv2D(filters, (1, 1), strides=stride, use_bias=False,
                                   name="res{}{}_branch2a".format(stage_char, block_char), **parameters)(x)
        y = layers.BatchNormalization(axis=axis, epsilon=1e-5,
                                               name="bn{}{}_branch2a".format(stage_char, block_char))(y)
        y = layers.Activation("relu", name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

        y = layers.ZeroPadding2D(padding=1, name="padding{}{}_branch2b".format(stage_char, block_char))(y)
        y = layers.Conv2D(filters, kernel_size, use_bias=False,
                                   name="res{}{}_branch2b".format(stage_char, block_char), **parameters)(y)
        y = layers.BatchNormalization(axis=axis, epsilon=1e-5,
                                               name="bn{}{}_branch2b".format(stage_char, block_char))(y)
        y = layers.Activation("relu", name="res{}{}_branch2b_relu".format(stage_char, block_char))(y)

        y = layers.Conv2D(filters * 4, (1, 1), use_bias=False,
                                   name="res{}{}_branch2c".format(stage_char, block_char), **parameters)(y)
        y = layers.BatchNormalization(axis=axis, epsilon=1e-5,
                                               name="bn{}{}_branch2c".format(stage_char, block_char))(y)

        if block == 0:
            shortcut = layers.Conv2D(filters * 4, (1, 1), strides=stride, use_bias=False,
                                              name="res{}{}_branch1".format(stage_char, block_char), **parameters)(x)
            shortcut = layers.BatchNormalization(axis=axis, epsilon=1e-5,
                                                          name="bn{}{}_branch1".format(stage_char, block_char))(
                shortcut)
        else:
            shortcut = x

        y = layers.Add(name="res{}{}".format(stage_char, block_char))([y, shortcut])
        y = layers.Activation("relu", name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f
