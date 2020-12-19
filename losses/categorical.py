from typing import List

import tensorflow as tf
from ..utils.tensor import remove_indices


def soft_dice_coefficient(y_true, y_pred, exclude_classes: List[int] = None):
    """Computes the mean Soft Dice Coefficient (DSC)

    Mean Soft Dice Coefficient is a common evaluation metric for semantic image
    segmentation, which first computes the DSC for each semantic class and then
    computes the average over classes. DSC is defined as follows:
      DSC = (2 * true_positive) / (2 * true_positive + false_positive + false_negative).

    """

    axis = (1, 2)
    eps = tf.keras.backend.epsilon()
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    channel_dsc = (2.0 * intersection + eps) / (
        tf.reduce_sum(y_true, axis=axis) + tf.reduce_sum(y_pred, axis=axis) + eps
    )

    if exclude_classes is not None:
        channel_dsc = remove_indices(channel_dsc, exclude_classes, axis=1)

    sample_dsc = tf.reduce_mean(channel_dsc, axis=1)
    batch_dsc = tf.reduce_mean(sample_dsc, axis=0)
    return 1 - batch_dsc
