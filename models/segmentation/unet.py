import tensorflow as tf
from tensorflow.python.keras import layers

def UNet(inputs, num_classes):

    with tf.name_scope("UNet"):
        # Downsampling Path
        conv1 = layers.Conv2D(16, kernel_size=(3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
        conv1 = layers.Conv2D(16, kernel_size=(3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = layers.Conv2D(32, kernel_size=(3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool1)
        conv2 = layers.Conv2D(32, kernel_size=(3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = layers.Conv2D(64, kernel_size=(3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool2)
        conv3 = layers.Conv2D(64, kernel_size=(3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = layers.Conv2D(128, kernel_size=(3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool3)
        conv4 = layers.Conv2D(128, kernel_size=(3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv4)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = layers.Conv2D(256, kernel_size=(3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool4)
        conv5 = layers.Conv2D(256, kernel_size=(3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv5)

        # Upsampling Path
        upconv1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
        upconv1 = layers.concatenate([upconv1, conv4])
        conv6 = layers.Conv2D(128, kernel_size=(3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(upconv1)
        conv6 = layers.Conv2D(128, kernel_size=(3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv6)

        upconv2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
        upconv2 = layers.concatenate([upconv2, conv3])
        conv7 = layers.Conv2D(64, kernel_size=(3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(upconv2)
        conv7 = layers.Conv2D(64, kernel_size=(3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv7)

        upconv3 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7)
        upconv3 = layers.concatenate([upconv3, conv2])
        conv8 = layers.Conv2D(32, kernel_size=(3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(upconv3)
        conv8 = layers.Conv2D(32, kernel_size=(3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv8)

        upconv4 = layers.Conv2DTranspose(16, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv8)
        upconv4 = layers.concatenate([upconv4, conv1], axis=3)
        conv9 = layers.Conv2D(16, kernel_size=(3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(upconv4)
        conv9 = layers.Conv2D(16, kernel_size=(3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv9)
        #conv9 = layers.SpatialDropout2D
        conv9 = layers.Conv2D(num_classes, kernel_size=(1, 1))(conv9)
        outputs = layers.Softmax(name="softmax_out")(conv9)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs], name="UNet")

    return model