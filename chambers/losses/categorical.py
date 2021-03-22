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


def _arcface_logits(y_true, y_pred, cos_m, sin_m, mm, threshold, scale=64.0):
    cos_t = y_pred
    sin_t = tf.sqrt(1.0 - cos_t ** 2)

    cos_mt = (cos_t * cos_m) - (sin_t * sin_m)
    cos_mt = tf.where(cos_t > threshold, cos_mt, cos_t - mm)

    y_true = tf.cast(y_true, y_pred.dtype)
    logits = (cos_mt * y_true) + (cos_t * (1 - y_true))
    logits = logits * scale
    return logits


class ArcFace(tf.keras.losses.CategoricalCrossentropy):
    def __init__(
        self,
        scale=64.0,
        margin=0.5,
        reduction=tf.keras.losses.Reduction.AUTO,
        name=None,
    ):
        super(ArcFace, self).__init__(from_logits=True, reduction=reduction, name=name)
        self.scale = scale
        self.margin = margin

        self._cos_m = tf.cos(margin)
        self._sin_m = tf.sin(margin)
        self._mm = self._sin_m * margin
        self._threshold = tf.cos(3.141592653589793 - margin)

    def call(self, y_true, y_pred):
        logits = _arcface_logits(
            y_true,
            y_pred,
            cos_m=self._cos_m,
            sin_m=self._sin_m,
            mm=self._mm,
            threshold=self._threshold,
            scale=self.scale,
        )
        return super(ArcFace, self).call(y_true, logits)


class SparseArcFace(tf.keras.losses.SparseCategoricalCrossentropy):
    def __init__(
        self,
        scale=64.0,
        margin=0.5,
        reduction=tf.keras.losses.Reduction.AUTO,
        name=None,
    ):
        super(SparseArcFace, self).__init__(
            from_logits=True, reduction=reduction, name=name
        )
        self.scale = scale
        self.margin = margin

        self._cos_m = tf.cos(margin)
        self._sin_m = tf.sin(margin)
        self._mm = self._sin_m * margin
        self._threshold = tf.cos(3.141592653589793 - margin)

    def call(self, y_true, y_pred):
        n_classes = tf.shape(y_pred)[-1]
        y_true_1hot = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_1hot = tf.one_hot(y_true_1hot, depth=n_classes)
        logits = _arcface_logits(
            y_true_1hot,
            y_pred,
            cos_m=self._cos_m,
            sin_m=self._sin_m,
            mm=self._mm,
            threshold=self._threshold,
            scale=self.scale,
        )
        return super(SparseArcFace, self).call(y_true, logits)