from typing import List

import tensorflow as tf
from tensorflow.python.keras.metrics import MeanMetricWrapper

from .losses import soft_dice_coefficient as _dsc


class F1(tf.keras.metrics.Metric):
    def __init__(self, thresholds=None, top_k=None, class_id=None, name=None, dtype=None):
        super(F1, self).__init__(name=name, dtype=dtype)
        self.thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id
        self.precision = tf.keras.metrics.Precision(thresholds=self.thresholds,
                                                    top_k=self.top_k,
                                                    class_id=self.class_id,
                                                    name=name,
                                                    dtype=dtype
                                                    )
        self.recall = tf.keras.metrics.Recall(thresholds=self.thresholds,
                                              top_k=self.top_k,
                                              class_id=self.class_id,
                                              name=name,
                                              dtype=dtype
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
            'thresholds': self.thresholds,
            'top_k': self.top_k,
            'class_id': self.class_id
        }
        base_config = super(F1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class IntersectionOverUnion(tf.keras.metrics.Metric):
    """Computes the mean Intersection-Over-Union metric.
    Mean Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation, which first computes the IOU for each semantic class and then
    computes the average over classes. IOU is defined as follows:
      IOU = true_positive / (true_positive + false_positive + false_negative).
    The predictions are accumulated in a confusion matrix, weighted by
    `sample_weight` and the metric is then calculated from it.
    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.
    Usage:
    ```python
    m = tf.keras.metrics.MeanIoU(num_classes=2)
    m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
      # cm = [[1, 1],
              [1, 1]]
      # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
      # iou = true_positives / (sum_row + sum_col - true_positives))
      # result = (1 / (2 + 2 - 1) + 1 / (2 + 2 - 1)) / 2 = 0.33
    print('Final result: ', m.result().numpy())  # Final result: 0.33
    ```
    Usage with tf.keras API:
    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile(
      'sgd',
      loss='mse',
      metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
    ```
    """

    def __init__(self, num_classes: int, exclude_classes: List[int] = None, onehot_encoded: bool = False,
                 name: str = "intersection_over_union", dtype=None):
        """Creates a `IntersectionOverUnion` instance.
        Args:
          num_classes: The possible number of labels the prediction task can have.
            This value must be provided, since a confusion matrix of dimension =
            [num_classes, num_classes] will be allocated.
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
        """
        super().__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.exclude_classes = exclude_classes
        self.onehot_encoded = onehot_encoded

        if self.exclude_classes is not None:
            indices = tf.expand_dims(tf.convert_to_tensor(exclude_classes, dtype=tf.int32), -1)
            updates = tf.zeros_like(exclude_classes, dtype=tf.bool)
            self._class_mask = tf.ones([self.num_classes], dtype=tf.bool)
            self._class_mask = tf.tensor_scatter_nd_update(self._class_mask, indices, updates)

        # Variable to accumulate the predictions in the confusion matrix. Setting
        # the type to be `float64` as required by confusion_matrix_ops.
        self.total_cm = self.add_weight(
            'total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer=tf.zeros_initializer,
            dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.
        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.
        Returns:
          Update op.
        """

        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)

        # Find class index with highest predicted probability, if inputs are onehot encoded
        if self.onehot_encoded:
            y_true = tf.argmax(y_true, -1)
            y_pred = tf.argmax(y_pred, -1)

        # Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])

        if sample_weight is not None and sample_weight.shape.ndims > 1:
            sample_weight = tf.reshape(sample_weight, [-1])

        # Accumulate the prediction to current confusion matrix.
        current_cm = tf.math.confusion_matrix(
            y_true,
            y_pred,
            self.num_classes,
            weights=sample_weight,
            dtype=tf.float64)
        return self.total_cm.assign_add(current_cm)

    def result(self):
        """Compute the mean intersection-over-union via the confusion matrix."""
        sum_over_row = tf.cast(
            tf.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = tf.cast(
            tf.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = tf.cast(
            tf.linalg.diag_part(self.total_cm), dtype=self._dtype)

        if self.exclude_classes is not None:
            sum_over_row = sum_over_row[self._class_mask]
            sum_over_col = sum_over_col[self._class_mask]
            true_positives = true_positives[self._class_mask]

        false_positives = sum_over_col - true_positives
        false_negatives = sum_over_row - true_positives

        denominator = true_positives + false_positives + false_negatives
        iou = tf.math.divide_no_nan(true_positives, denominator)

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = tf.reduce_sum(
            tf.cast(tf.not_equal(denominator, 0), dtype=self._dtype))

        return tf.math.divide_no_nan(
            tf.reduce_sum(iou, name='mean_iou'), num_valid_entries)

    def reset_states(self):
        self.total_cm.assign(tf.zeros_like(self.total_cm))

    def get_config(self):
        config = {'num_classes': self.num_classes, "exclude_classes": self.class_mask,
                  "onehot_encoded": self.onehot_encoded}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SoftDiceCoefficient(MeanMetricWrapper):
    def __init__(self, exclude_classes: List[int] = None, name="soft_dice_coefficient", dtype=None):
        super().__init__(soft_dice_coefficient, name=name, dtype=dtype, exclude_classes=exclude_classes)

    def get_config(self):
        config = {"exclude_classes": self.class_mask}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


# from .utils.ranking_utils import score_matrix_to_binary_ranking
#
#
# class GlobalMeanAveragePrecision(tf.keras.metrics.Metric):
#     def __init__(self, max_samples, dim, name="ranking_metrics"):
#         super().__init__(name=name)
#         self.max_samples = max_samples
#         self.dim = dim
#         self.y_preds = self.add_weight(shape=(max_samples, dim), name="y_preds")
#         self.y_trues = self.add_weight(shape=(max_samples, 1), name="y_trues")
#         self.n_trues = self.add_weight(name="n_trues", dtype=tf.int64, initializer="zeros")
#         self.n_preds = self.add_weight(name="n_preds", dtype=tf.int64, initializer="zeros")
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         n_true = tf.cast(tf.shape(y_true)[0], tf.int64)
#         n_pred = tf.cast(tf.shape(y_pred)[0], tf.int64)
#
#         tf.print(n_true)
#         tf.print(n_pred)
#
#         true_current = self._get_current_trues()
#         true_fill = tf.zeros((self.max_samples - (self.n_trues + n_true), 1))
#         self.y_trues.assign(tf.concat([true_current, true_fill], axis=0))
#
#         pred_current = self._get_current_preds()
#         pred_fill = tf.zeros((self.max_samples - (self.n_preds + n_pred), self.dim))
#         self.y_preds.assign(tf.concat([pred_current, pred_fill], axis=0))
#
#         self.n_trues.assign_add(n_true)
#         self.n_preds.assign_add(n_pred)
#
#     def reset_states(self):
#         self.y_trues.assign(tf.zeros_like(self.y_trues))
#         self.y_preds.assign(tf.zeros_like(self.y_preds))
#
#     def result(self, k=10):
#         y_trues = self._get_current_trues()
#         y_preds = self._get_current_preds()
#
#         score_mat = tf.matmul(y_preds, tf.transpose(y_preds))
#         score_mat = score_matrix_to_binary_ranking(score_mat, y_trues, remove_diag=True)
#         return mean_average_precision(score_mat, k)
#
#     def _get_current_trues(self):
#         return tf.slice(self.y_trues, [0, 0], [self.n_trues, -1])
#
#     def _get_current_preds(self):
#         return tf.slice(self.y_preds, [0, 0], [self.n_preds, -1])


class RankingAccuracy:

    def __call__(self, binary_ranking):
        return ranking_accuracy(binary_ranking)

    @property
    def __name__(self):
        return "r_acc"


class RecallAtK:
    def __init__(self, k: int):
        self.k = k

    def __call__(self, binary_ranking):
        return recall_at_k(binary_ranking, self.k)

    @property
    def __name__(self):
        return "r@{}".format(self.k)


class PrecisionAtK:
    def __init__(self, k: int):
        self.k = k

    def __call__(self, binary_ranking):
        return precision_at_k(binary_ranking, self.k)

    @property
    def __name__(self):
        return "p@{}".format(self.k)


class MeanAveragePrecisionAtK:
    def __init__(self, k: int):
        self.k = k

    def __call__(self, binary_ranking):
        return mean_average_precision(binary_ranking, self.k)

    @property
    def __name__(self):
        return "map@{}".format(self.k)


class MeanAveragePrecision:

    def __call__(self, binary_ranking):
        return mean_average_precision(binary_ranking, None)

    @property
    def __name__(self):
        return "map"


@tf.function
def ranking_accuracy(binary_ranking):
    n_true_pos = tf.cast(tf.reduce_sum(binary_ranking, axis=1), tf.int32)
    nrows = tf.shape(binary_ranking)[0]
    accs = tf.TensorArray(dtype=tf.float32, size=nrows, element_shape=tf.TensorShape(dims=()))
    for i in tf.range(nrows):
        ki = tf.slice(n_true_pos, [i], [1])[0]
        n_pred_pos = tf.reduce_sum(tf.slice(binary_ranking, [i, 0], [1, ki]))
        acc = n_pred_pos / tf.cast(ki, tf.float32)
        accs = accs.write(i, acc)
    return tf.reduce_mean(accs.stack())


def recall_at_k(binary_ranking, k: int):
    return tf.reduce_mean(tf.cast(tf.reduce_sum(binary_ranking[:, :k], axis=1) > 0, tf.float32))


def precision_at_k(binary_ranking, k: int):
    precisions_at_k = tf.reduce_sum(binary_ranking[:, :k], axis=1) / k
    return tf.reduce_mean(precisions_at_k)


def mean_average_precision(binary_ranking, k: int = None):
    if k is None:
        k = tf.shape(binary_ranking)[1]

    binary_ranking = binary_ranking[:, :k]
    n_positive_predictions = tf.reduce_sum(binary_ranking, axis=1)
    rank_out_of_k = tf.range(1, k + 1, dtype=tf.float32)

    cumulative_positive_at_k = tf.cumsum(binary_ranking, axis=1) * binary_ranking
    k_precisions = tf.math.divide_no_nan(cumulative_positive_at_k, rank_out_of_k)
    average_precision_at_k = tf.math.divide_no_nan(tf.reduce_sum(k_precisions, axis=1), n_positive_predictions)
    return tf.reduce_mean(average_precision_at_k)


def soft_dice_coefficient(y_true, y_pred, exclude_classes: List[int] = None):
    return tf.abs(_dsc(y_true, y_pred, exclude_classes=exclude_classes) - 1)


# Aliases
map = MAP = mean_average_precision
dsc = DSC = soft_dice_coefficient
