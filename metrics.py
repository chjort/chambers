import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.metrics import Metric, MeanMetricWrapper


class MeanIoU(Metric):
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

    def __init__(self, num_classes, onehot_encoded=False, name=None, dtype=None):
        """Creates a `MeanIoU` instance.

        Args:
          num_classes: The possible number of labels the prediction task can have.
            This value must be provided, since a confusion matrix of dimension =
            [num_classes, num_classes] will be allocated.
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
        """
        super(MeanIoU, self).__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.onehot_encoded = onehot_encoded

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
        K.set_value(self.total_cm, np.zeros((self.num_classes, self.num_classes)))

    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super(MeanIoU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MeanDSC(MeanMetricWrapper):
    """Computes the mean Soft Dice Coefficient

    Mean Soft Dice Coefficient is a common evaluation metric for semantic image
    segmentation, which first computes the DSC for each semantic class and then
    computes the average over classes. DSC is defined as follows:
      DSC = (2 * true_positive) / (2 * true_positive + false_positive + false_negative).
    The predictions are accumulated in a confusion matrix, weighted by
    `sample_weight` and the metric is then calculated from it.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Usage:

    ```python
    m = tf.keras.metrics.MeanDSC(axis=None)
    m.update_state([0., 1., 0., 0.], [0., 0.85, 0.15, 0.])
    print('Final result: ', m.result().numpy())  # Final result: 0.85
    ```

    Usage with tf.keras API:

    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile(
      'sgd',
      loss='mse',
      metrics=[tf.keras.metrics.MeanDSC(axis)])
    ```
    """

    def __init__(self, axis=None, name="soft_dice_coefficient", dtype=None):
        super(MeanDSC, self).__init__(soft_dice_coefficient, name=name, dtype=dtype, axis=axis)


class MeanDHC(MeanMetricWrapper):
    """Computes the mean Hard Dice Coefficient
    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Usage:

    ```python
    m = tf.keras.metrics.MeanDHC()
    m.update_state([0., 1., 0., 0.], [0., 0.85, 0.15, 0.])
    print('Final result: ', m.result().numpy())  # Final result: 1
    ```

    Usage with tf.keras API:

    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile(
      'sgd',
      loss='mse',
      metrics=[tf.keras.metrics.MeanDHC()])
    ```
    """

    def __init__(self, axis=None, threshold=0.5, name="hard_dice_coefficient", dtype=None):
        super(MeanDHC, self).__init__(hard_dice_coefficient, name=name, dtype=dtype, axis=axis, threshold=threshold)


class F1(Metric):
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


def soft_dice_coefficient(y_true, y_pred, axis=None):
    if axis==None:
        rank = y_true.shape.rank
        if rank == 1:
            axis = 0
        else:
            axis = tuple(i for i in range(1, rank))

    smooth = 1e-5
    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    batch_dsc = (2. * intersection + smooth) / (tf.reduce_sum(y_true, axis=axis) + tf.reduce_sum(y_pred, axis=axis) + smooth)
    return batch_dsc


def hard_dice_coefficient(y_true, y_pred, threshold, axis=None):
    y_true = tf.cast(y_true > threshold, dtype=tf.float32)
    y_pred = tf.cast(y_pred > threshold, dtype=tf.float32)
    return soft_dice_coefficient(y_true, y_pred, axis=axis)