import tensorflow as tf


def inputs_to_input_layer(input_tensor=None, input_shape=None, name=None):
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape=input_shape, name=name)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape, name=name)
        else:
            img_input = input_tensor

    return img_input
