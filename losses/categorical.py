import tensorflow as tf
from typing import List


def soft_dice_coefficient(y_true, y_pred, channel_mask: List[bool] = None):
    axis = (1, 2)
    eps = tf.keras.backend.epsilon()
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    channel_dsc = (2. * intersection + eps) / (
            tf.reduce_sum(y_true, axis=axis) + tf.reduce_sum(y_pred, axis=axis) + eps)

    if channel_mask is not None:
        channel_mask = tf.convert_to_tensor(channel_mask)
        indices = tf.range(tf.shape(channel_dsc)[1])
        channel_dsc = tf.gather(channel_dsc, indices[channel_mask], axis=1)

    sample_dsc = tf.reduce_mean(channel_dsc, axis=1)
    batch_dsc = tf.reduce_mean(sample_dsc, axis=0)
    return 1 - batch_dsc


def hard_dice_coefficient(y_true, y_pred, channel_mask: List[bool] = None):
    y_true_h = tf.round(y_true)
    y_pred_h = tf.round(y_pred)
    return soft_dice_coefficient(y_true_h, y_pred_h, channel_mask=channel_mask)
