import tensorflow as tf
from chambers.models import ResNet50, ResNet101, ResNet152


def conv_bn_relu(input, num_channel, kernel_size, stride, name, padding='same', bn_axis=-1, bn_momentum=0.99,
                 bn_scale=True, use_bias=True):
    x = tf.keras.layers.Conv2D(filters=num_channel, kernel_size=(kernel_size, kernel_size),
               strides=stride, padding=padding,
               kernel_initializer="he_normal",
               use_bias=use_bias,
               name=name + "_conv")(input)
    x = tf.keras.layers.BatchNormalization(name=name + '_bn', scale=bn_scale, axis=bn_axis, momentum=bn_momentum, epsilon=1.001e-5, )(x)
    x = tf.keras.layers.Activation('relu', name=name + '_relu')(x)
    return x


def conv_bn(input, num_channel, kernel_size, stride, name, padding='same', bn_axis=-1, bn_momentum=0.99, bn_scale=True,
            use_bias=True):
    x = tf.keras.layers.Conv2D(filters=num_channel, kernel_size=(kernel_size, kernel_size),
               strides=stride, padding=padding,
               kernel_initializer="he_normal",
               use_bias=use_bias,
               name=name + "_conv")(input)
    x = tf.keras.layers.BatchNormalization(name=name + '_bn', scale=bn_scale, axis=bn_axis, momentum=bn_momentum, epsilon=1.001e-5, )(x)
    return x


def conv_relu(input, num_channel, kernel_size, stride, name, padding='same', use_bias=True, activation='relu'):
    x = tf.keras.layers.Conv2D(filters=num_channel, kernel_size=(kernel_size, kernel_size),
               strides=stride, padding=padding,
               kernel_initializer="he_normal",
               use_bias=use_bias,
               name=name + "_conv")(input)
    x = tf.keras.layers.Activation(activation, name=name + '_relu')(x)
    return x


def decoder_block(input, filters, skip, block_name):
    x = tf.keras.layers.UpSampling2D()(input)
    x = conv_bn_relu(x, filters, 3, stride=1, padding='same', name=block_name + '_conv1')
    x = tf.keras.layers.concatenate([x, skip], axis=-1, name=block_name + '_concat')
    x = conv_bn_relu(x, filters, 3, stride=1, padding='same', name=block_name + '_conv2')
    return x


def decoder_block_no_bn(input, filters, skip, block_name, activation='relu'):
    x = tf.keras.layers.UpSampling2D()(input)
    x = conv_relu(x, filters, 3, stride=1, padding='same', name=block_name + '_conv1', activation=activation)
    x = tf.keras.layers.concatenate([x, skip], axis=-1, name=block_name + '_concat')
    x = conv_relu(x, filters, 3, stride=1, padding='same', name=block_name + '_conv2', activation=activation)
    return x



def create_pyramid_features(C1, C2, C3, C4, C5, feature_size=256):
    P5 = tf.keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='P5', kernel_initializer="he_normal")(C5)
    P5_upsampled = tf.keras.layers.UpSampling2D(name='P5_upsampled')(P5)

    P4 = tf.keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced',
                kernel_initializer="he_normal")(C4)
    P4 = tf.keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4 = tf.keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4', kernel_initializer="he_normal")(P4)
    P4_upsampled = tf.keras.layers.UpSampling2D(name='P4_upsampled')(P4)

    P3 = tf.keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced',
                kernel_initializer="he_normal")(C3)
    P3 = tf.keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = tf.keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3', kernel_initializer="he_normal")(P3)
    P3_upsampled = tf.keras.layers.UpSampling2D(name='P3_upsampled')(P3)

    P2 = tf.keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C2_reduced',
                kernel_initializer="he_normal")(C2)
    P2 = tf.keras.layers.Add(name='P2_merged')([P3_upsampled, P2])
    P2 = tf.keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P2', kernel_initializer="he_normal")(P2)
    P2_upsampled = tf.keras.layers.UpSampling2D(size=(2, 2), name='P2_upsampled')(P2)

    P1 = tf.keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C1_reduced',
                kernel_initializer="he_normal")(C1)
    P1 = tf.keras.layers.Add(name='P1_merged')([P2_upsampled, P1])
    P1 = tf.keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P1', kernel_initializer="he_normal")(P1)

    return P1, P2, P3, P4, P5


def prediction_fpn_block(x, name, upsample=None):
    x = conv_relu(x, 128, 3, stride=1, name="predcition_" + name + "_1")
    x = conv_relu(x, 128, 3, stride=1, name="prediction_" + name + "_2")
    if upsample:
        x = tf.keras.layers.UpSampling2D(upsample)(x)
    return x


def ResNet50_FPN(inputs, num_classes, weights="imagenet", activation="softmax"):
    resnet_base = ResNet50(inputs, weights=weights, include_top=True)
    conv1 = resnet_base.get_layer("conv1_relu").output
    conv2 = resnet_base.get_layer("res2c_relu").output
    conv3 = resnet_base.get_layer("res3d_relu").output
    conv4 = resnet_base.get_layer("res4f_relu").output
    conv5 = resnet_base.get_layer("res5c_relu").output
    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = tf.keras.layers.concatenate(
        [
            prediction_fpn_block(P5, "P5", (8, 8)),
            prediction_fpn_block(P4, "P4", (4, 4)),
            prediction_fpn_block(P3, "P3", (2, 2)),
            prediction_fpn_block(P2, "P2"),
        ]
    )
    x = conv_bn_relu(x, 256, 3, (1, 1), name="aggregation")
    x = decoder_block_no_bn(x, 128, conv1, 'up4')
    x = tf.keras.layers.UpSampling2D()(x)
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv1")
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv2")
    if activation == 'softmax':
        name = 'mask_softmax'
        x = tf.keras.layers.Conv2D(num_classes, (1, 1), activation=activation, name=name)(x)
    else:
        x = tf.keras.layers.Conv2D(num_classes, (1, 1), activation=activation, name="mask")(x)
    model = tf.keras.models.Model(inputs, x, name="ResNet50_FPN")
    return model


def ResNet101_FPN(inputs, num_classes, weights="imagenet", activation="softmax"):
    resnet_base = ResNet101(inputs, weights=weights, include_top=True)
    conv1 = resnet_base.get_layer("conv1_relu").output
    conv2 = resnet_base.get_layer("res2c_relu").output
    conv3 = resnet_base.get_layer("res3b3_relu").output
    conv4 = resnet_base.get_layer("res4b22_relu").output
    conv5 = resnet_base.get_layer("res5c_relu").output
    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = tf.keras.layers.concatenate(
        [
            prediction_fpn_block(P5, "P5", (8, 8)),
            prediction_fpn_block(P4, "P4", (4, 4)),
            prediction_fpn_block(P3, "P3", (2, 2)),
            prediction_fpn_block(P2, "P2"),
        ]
    )
    x = conv_bn_relu(x, 256, 3, (1, 1), name="aggregation")
    x = decoder_block_no_bn(x, 128, conv1, 'up4')
    x = tf.keras.layers.UpSampling2D()(x)
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv1")
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv2")
    if activation == 'softmax':
        name = 'mask_softmax'
        x = tf.keras.layers.Conv2D(num_classes, (1, 1), activation=activation, name=name)(x)
    else:
        x = tf.keras.layers.Conv2D(num_classes, (1, 1), activation=activation, name="mask")(x)
    model = tf.keras.models.Model(inputs, x, name="ResNet101_FPN")
    return model


def ResNet152_FPN(inputs, num_classes, weights="imagenet", activation="softmax"):
    resnet_base = ResNet152(inputs, weights=weights, include_top=True)
    conv1 = resnet_base.get_layer("conv1_relu").output
    conv2 = resnet_base.get_layer("res2c_relu").output
    conv3 = resnet_base.get_layer("res3b7_relu").output
    conv4 = resnet_base.get_layer("res4b35_relu").output
    conv5 = resnet_base.get_layer("res5c_relu").output
    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = tf.keras.layers.concatenate(
        [
            prediction_fpn_block(P5, "P5", (8, 8)),
            prediction_fpn_block(P4, "P4", (4, 4)),
            prediction_fpn_block(P3, "P3", (2, 2)),
            prediction_fpn_block(P2, "P2"),
        ]
    )
    x = conv_bn_relu(x, 256, 3, (1, 1), name="aggregation")
    x = decoder_block_no_bn(x, 128, conv1, 'up4')
    x = tf.keras.layers.UpSampling2D()(x)
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv1")
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv2")
    x = tf.keras.layers.Conv2D(num_classes, (1, 1), name="mask", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Activation(activation)(x)
    model = tf.keras.models.Model(inputs, x, name="ResNet152_FPN")
    return model
