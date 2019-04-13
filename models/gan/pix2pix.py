import tensorflow as tf
from tensorflow.python.keras import layers


def Generator(inputs):
    with tf.name_scope("Generator"):
        # Downsampling path
        conv1 = layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same",
                              kernel_initializer=tf.random_normal_initializer(0, 0.02), use_bias=False)(inputs)
        lrelu1 = layers.LeakyReLU()(conv1)

        conv2 = layers.Conv2D(64, (4, 4), (2, 2), "same", kernel_initializer=tf.random_normal_initializer(0, 0.02), use_bias=False)(lrelu1)
        bn2 = layers.BatchNormalization()(conv2)
        lrelu2 = layers.LeakyReLU()(bn2)

        conv3 = layers.Conv2D(128, (4, 4), (2, 2), "same", kernel_initializer=tf.random_normal_initializer(0, 0.02), use_bias=False)(lrelu2)
        bn3 = layers.BatchNormalization()(conv3)
        lrelu3 = layers.LeakyReLU()(bn3)

        conv4 = layers.Conv2D(256, (4, 4), (2, 2), "same", kernel_initializer=tf.random_normal_initializer(0, 0.02), use_bias=False)(lrelu3)
        bn4 = layers.BatchNormalization()(conv4)
        lrelu4 = layers.LeakyReLU()(bn4)

        conv5 = layers.Conv2D(512, (4, 4), (2, 2), "same", kernel_initializer=tf.random_normal_initializer(0, 0.02), use_bias=False)(lrelu4)
        bn5 = layers.BatchNormalization()(conv5)
        lrelu5 = layers.LeakyReLU()(bn5)

        conv6 = layers.Conv2D(512, (4, 4), (2, 2), "same", kernel_initializer=tf.random_normal_initializer(0, 0.02), use_bias=False)(lrelu5)
        bn6 = layers.BatchNormalization()(conv6)
        lrelu6 = layers.LeakyReLU()(bn6)

        conv7 = layers.Conv2D(512, (4, 4), (2, 2), "same", kernel_initializer=tf.random_normal_initializer(0, 0.02), use_bias=False)(lrelu6)
        bn7 = layers.BatchNormalization()(conv7)
        lrelu7 = layers.LeakyReLU()(bn7)

        conv8 = layers.Conv2D(512, (4, 4), (2, 2), "same", kernel_initializer=tf.random_normal_initializer(0, 0.02), use_bias=False)(lrelu7)
        bn8 = layers.BatchNormalization()(conv8)
        lrelu8 = layers.LeakyReLU()(bn8)

        conv9 = layers.Conv2D(512, (4, 4), (2, 2), "same", kernel_initializer=tf.random_normal_initializer(0, 0.02), use_bias=False)(lrelu8)
        bn9 = layers.BatchNormalization()(conv9)
        lrelu9 = layers.LeakyReLU()(bn9)


        # Upsampling path


# Generator
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def GeneratorOLD():
    OUTPUT_CHANNELS = 3

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 256, 256, 3)

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
