from typing import List

import tensorflow as tf
from tensorflow.python.keras.metrics import MeanMetricWrapper

from chambers.losses.categorical import soft_dice_coefficient as _dsc


@tf.keras.utils.register_keras_serializable(package="Chambers")
class F1(tf.keras.metrics.Metric):
    def __init__(
        self, thresholds=None, top_k=None, class_id=None, name=None, dtype=None
    ):
        super(F1, self).__init__(name=name, dtype=dtype)
        self.thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id
        self.precision = tf.keras.metrics.Precision(
            thresholds=self.thresholds,
            top_k=self.top_k,
            class_id=self.class_id,
            name=name,
            dtype=dtype,
        )
        self.recall = tf.keras.metrics.Recall(
            thresholds=self.thresholds,
            top_k=self.top_k,
            class_id=self.class_id,
            name=name,
            dtype=dtype,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight=sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

    def get_config(self):
        config = {
            "thresholds": self.thresholds,
            "top_k": self.top_k,
            "class_id": self.class_id,
        }
        base_config = super(F1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="Chambers")
class SoftDiceCoefficient(MeanMetricWrapper):
    def __init__(
        self,
        exclude_classes: List[int] = None,
        name="soft_dice_coefficient",
        dtype=None,
    ):
        super().__init__(
            soft_dice_coefficient,
            name=name,
            dtype=dtype,
            exclude_classes=exclude_classes,
        )

    def get_config(self):
        config = {"exclude_classes": self.class_mask}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def soft_dice_coefficient(y_true, y_pred, exclude_classes: List[int] = None):
    return tf.abs(_dsc(y_true, y_pred, exclude_classes=exclude_classes) - 1)


# Aliases
dsc = DSC = soft_dice_coefficient
