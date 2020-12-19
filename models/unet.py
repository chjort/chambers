import tensorflow as tf


def UNet_small(input_tensor, num_classes, hidden_activation="elu"):
    min_factor = 16
    # padding = pad_to_factor(input_tensor.shape.as_list()[1:3], min_factor)

    # padded_inputs = tf.keras.layers.ZeroPadding2D(padding)(input_tensor)
    with tf.name_scope("UNet"):
        # Downsampling Path
        conv1 = tf.keras.layers.Conv2D(
            16,
            kernel_size=(3, 3),
            activation=hidden_activation,
            kernel_initializer="he_normal",
            padding="same",
        )(input_tensor)
        conv1 = tf.keras.layers.Conv2D(
            16,
            kernel_size=(3, 3),
            activation=hidden_activation,
            kernel_initializer="he_normal",
            padding="same",
        )(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = tf.keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            activation=hidden_activation,
            kernel_initializer="he_normal",
            padding="same",
        )(pool1)
        conv2 = tf.keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            activation=hidden_activation,
            kernel_initializer="he_normal",
            padding="same",
        )(conv2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = tf.keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            activation=hidden_activation,
            kernel_initializer="he_normal",
            padding="same",
        )(pool2)
        conv3 = tf.keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            activation=hidden_activation,
            kernel_initializer="he_normal",
            padding="same",
        )(conv3)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = tf.keras.layers.Conv2D(
            128,
            kernel_size=(3, 3),
            activation=hidden_activation,
            kernel_initializer="he_normal",
            padding="same",
        )(pool3)
        conv4 = tf.keras.layers.Conv2D(
            128,
            kernel_size=(3, 3),
            activation=hidden_activation,
            kernel_initializer="he_normal",
            padding="same",
        )(conv4)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = tf.keras.layers.Conv2D(
            256,
            kernel_size=(3, 3),
            activation=hidden_activation,
            kernel_initializer="he_normal",
            padding="same",
        )(pool4)
        conv5 = tf.keras.layers.Conv2D(
            256,
            kernel_size=(3, 3),
            activation=hidden_activation,
            kernel_initializer="he_normal",
            padding="same",
        )(conv5)

        # Upsampling Path
        upconv1 = tf.keras.layers.Conv2DTranspose(
            128, (2, 2), strides=(2, 2), padding="same"
        )(conv5)
        upconv1 = tf.keras.layers.concatenate([upconv1, conv4])
        conv6 = tf.keras.layers.Conv2D(
            128,
            kernel_size=(3, 3),
            activation=hidden_activation,
            kernel_initializer="he_normal",
            padding="same",
        )(upconv1)
        conv6 = tf.keras.layers.Conv2D(
            128,
            kernel_size=(3, 3),
            activation=hidden_activation,
            kernel_initializer="he_normal",
            padding="same",
        )(conv6)

        upconv2 = tf.keras.layers.Conv2DTranspose(
            64, (2, 2), strides=(2, 2), padding="same"
        )(conv6)
        upconv2 = tf.keras.layers.concatenate([upconv2, conv3])
        conv7 = tf.keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            activation=hidden_activation,
            kernel_initializer="he_normal",
            padding="same",
        )(upconv2)
        conv7 = tf.keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            activation=hidden_activation,
            kernel_initializer="he_normal",
            padding="same",
        )(conv7)

        upconv3 = tf.keras.layers.Conv2DTranspose(
            32, (2, 2), strides=(2, 2), padding="same"
        )(conv7)
        upconv3 = tf.keras.layers.concatenate([upconv3, conv2])
        conv8 = tf.keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            activation=hidden_activation,
            kernel_initializer="he_normal",
            padding="same",
        )(upconv3)
        conv8 = tf.keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            activation=hidden_activation,
            kernel_initializer="he_normal",
            padding="same",
        )(conv8)

        upconv4 = tf.keras.layers.Conv2DTranspose(
            16, kernel_size=(2, 2), strides=(2, 2), padding="same"
        )(conv8)
        upconv4 = tf.keras.layers.concatenate([upconv4, conv1], axis=3)
        conv9 = tf.keras.layers.Conv2D(
            16,
            kernel_size=(3, 3),
            activation=hidden_activation,
            kernel_initializer="he_normal",
            padding="same",
        )(upconv4)
        conv9 = tf.keras.layers.Conv2D(
            16,
            kernel_size=(3, 3),
            activation=hidden_activation,
            kernel_initializer="he_normal",
            padding="same",
        )(conv9)
        # conv9 = tf.keras.layers.SpatialDropout2D
        conv9 = tf.keras.layers.Conv2D(num_classes, kernel_size=(1, 1))(conv9)
        softmax = tf.keras.layers.Softmax(name="softmax_out")(conv9)
    outputs = softmax
    # outputs = tf.keras.layers.Cropping2D(padding)(softmax)

    inputs = tf.keras.utils.get_source_inputs(input_tensor)
    model = tf.keras.models.Model(inputs=inputs, outputs=[outputs], name="UNet_small")

    return model
