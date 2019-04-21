import tensorflow as tf
from tensorflow.python.keras import layers
import numpy as np
import time

# class Conv2DBlock_LAYER(layers.Layer):
#     # TODO Fix parameters in Summary() for this layer
#     def __init__(self, filters, kernel_size, strides, use_batch_norm=True, padding="same", use_bias=False):
#         super(Conv2DBlock_LAYER, self).__init__()
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.strides = strides
#         self.use_batch_norm = use_batch_norm
#         self.padding = padding
#         self.use_bias = use_bias
#
#     def build(self, input_shape):
#         initializer = tf.random_normal_initializer(0, 0.02)
#         self.conv = layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
#                                   padding=self.padding, kernel_initializer=initializer, use_bias=self.use_bias
#                                   )
#         if self.use_batch_norm:
#             self.batch_norm = layers.BatchNormalization()
#         self.lrelu = layers.LeakyReLU()
#
#     def call(self, input, **kwargs):
#         x = self.conv(input)
#         if self.use_batch_norm:
#             x = self.batch_norm(x)
#         x = self.lrelu(x)
#         return x


def Conv2DBlock(x, filters, kernel_size=(4, 4), strides=(2, 2), use_batch_norm=True, padding="same", use_bias=False):
    initializer = tf.random_normal_initializer(0, 0.02)
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                      padding=padding, kernel_initializer=initializer, use_bias=use_bias
                      )(x)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    return x


def Conv2DTransposeBlock(x, filters, kernel_size=(4, 4), strides=(2, 2), padding="same",
                         use_bias=False, use_batch_norm=True, dropout=0.0):
    initializer = tf.random_normal_initializer(0, 0.02)
    x = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                               padding=padding, kernel_initializer=initializer, use_bias=use_bias
                               )(x)
    if use_batch_norm:
        x = layers.BatchNormalization(epsilon=1e-5, momentum=0.1,
                                      gamma_initializer=tf.random_normal_initializer(1, 0.02))(x)
    if dropout > 0.0:
        x = layers.Dropout(rate=dropout)(x)
    x = layers.ReLU()(x)

    return x


def Generator(inputs):
    with tf.name_scope("Generator"):
        # Downsampling path
        conv1 = Conv2DBlock(inputs, 64, use_batch_norm=False)
        conv2 = Conv2DBlock(conv1, 128)
        conv3 = Conv2DBlock(conv2, 256)
        conv4 = Conv2DBlock(conv3, 512)
        conv5 = Conv2DBlock(conv4, 512)
        conv6 = Conv2DBlock(conv5, 512)
        conv7 = Conv2DBlock(conv6, 512)
        conv8 = Conv2DBlock(conv7, 512)

        # Upsampling path
        tconv1 = Conv2DTransposeBlock(conv8, 512, dropout=0.5)
        concat1 = layers.Concatenate()([tconv1, conv7])

        tconv2 = Conv2DTransposeBlock(concat1, 512, dropout=0.5)
        concat2 = layers.Concatenate()([tconv2, conv6])

        tconv3 = Conv2DTransposeBlock(concat2, 512, dropout=0.5)
        concat3 = layers.Concatenate()([tconv3, conv5])

        tconv4 = Conv2DTransposeBlock(concat3, 512, )
        concat4 = layers.Concatenate()([tconv4, conv4])

        tconv5 = Conv2DTransposeBlock(concat4, 256)
        concat5 = layers.Concatenate()([tconv5, conv3])

        tconv6 = Conv2DTransposeBlock(concat5, 128)
        concat6 = layers.Concatenate()([tconv6, conv2])

        tconv7 = Conv2DTransposeBlock(concat6, 64)
        concat7 = layers.Concatenate()([tconv7, conv1])

    # output RGB image (3 channels)
    output = layers.Conv2DTranspose(3, kernel_size=(4, 4), strides=(2, 2),
                                    activation="tanh", padding="same",
                                    use_bias=False, kernel_initializer=tf.random_normal_initializer(0, 0.02))(concat7)

    return tf.keras.Model(inputs=inputs, outputs=output, name="Generator")


def Discriminator(inputs):
    outputs = layers.Input(shape=inputs.shape[1:])

    x = layers.Concatenate()([inputs, outputs])
    x = Conv2DBlock(x, 64, use_batch_norm=False)
    x = Conv2DBlock(x, 128)
    x = Conv2DBlock(x, 256)
    x = layers.ZeroPadding2D()(x)
    x = tf.keras.layers.Conv2D(512, (4, 4), padding="valid", kernel_initializer=tf.random_normal_initializer(0., 0.02),
                               use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.Conv2D(1, (4, 4), padding="valid", kernel_initializer=tf.random_normal_initializer(0., 0.02))(x)

    return tf.keras.Model(inputs=[inputs, outputs], outputs=x, name="Discriminator")


class Pix2Pix():
    def __init__(self, inputs):
        self.g = Generator(inputs)
        self.d = Discriminator(inputs)

        self.loss = tf.keras.losses.binary_crossentropy
        self.LAMBDA = 100
        self.g_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)
        self.d_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)

        self.train_step = tf.contrib.eager.defun(self._train_step)

    def discriminator_loss(self, y_d_real, y_d_gen):
        real_loss = self.loss(tf.ones_like(y_d_real), y_d_real)
        generated_loss = self.loss(tf.zeros_like(y_d_gen), y_d_gen)
        total_disc_loss = tf.reduce_mean(real_loss + generated_loss) #FIXME Reduce to mean or not?

        return total_disc_loss

    def generator_loss(self, y_d, y_g, y):
        gan_loss = tf.reduce_mean(self.loss(tf.ones_like(y_d), y_d)) #FIXME Reduce to mean or not?
        l1_loss = tf.reduce_mean(tf.abs(y - y_g))
        total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)

        return total_gen_loss

    def _train_step(self, X, y):
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            y_g = self.g(X, training=True)
            y_d_real = self.d([X, y], training=True)
            y_d_gen = self.d([X, y_g], training=True)

            g_loss = self.generator_loss(y_d_gen, y_g, y)
            d_loss = self.discriminator_loss(y_d_real, y_d_gen)

        g_gradients = g_tape.gradient(g_loss, self.g.trainable_variables)
        d_gradients = d_tape.gradient(d_loss, self.d.trainable_variables)

        self.g_optimizer.apply_gradients(zip(g_gradients, self.g.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_gradients, self.d.trainable_variables))

        return g_loss, d_loss

    def fit(self, dataset, epochs, steps_per_epoch):
        completed_epochs = 0
        g_epoch_losses = np.zeros(epochs)
        d_epoch_losses = np.zeros(epochs)

        g_step_losses = np.zeros(steps_per_epoch)
        d_step_losses = np.zeros(steps_per_epoch)

        start = time.time()
        for i, (X, Y) in enumerate(dataset):
            g_loss, d_loss = self.train_step(X, Y)

            j = i % (steps_per_epoch - 1)
            g_step_losses[j] = g_loss
            d_step_losses[j] = d_loss

            # print progress in steps of 5% training progression
            if (i + 1) % (steps_per_epoch // 20) == 0:
                print(g_step_losses[:j].mean(), d_step_losses[:j].mean())

            # if an epoch has passed
            if (i + 1) % steps_per_epoch == 0:
                completed_epochs += 1
                print("Epoch {} done. {}".format(completed_epochs, np.round(time.time()-start), 0))
                start = time.time()

                g_epoch_losses[completed_epochs - 1] = g_step_losses.mean()
                d_epoch_losses[completed_epochs - 1] = d_step_losses.mean()

                if completed_epochs == epochs:
                    print("Finished training")
                    return g_epoch_losses, d_epoch_losses

