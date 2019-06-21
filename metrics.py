import tensorflow as tf


def top_2_categorical_accuracy(y_true, y_pred):
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    return tf.keras.metrics.top_k_categorical_accuracy(y_true_flat, y_pred_flat, k=2)


def soft_dice_coef(y_true, y_pred):
    """


    :param y_true: Tensor of true target
    :param y_pred: Tensor of predictions
    :return: float scalar
    """
    smooth = 1e-5

    intersection = tf.reduce_sum(y_true * y_pred)
    batch_dsc = (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
    return batch_dsc


def soft_dice_coef_channelwise(y_true, y_pred):
    axis = (1, 2)
    smooth = 1e-5

    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    batch_dsc = (2. * intersection + smooth) / (tf.reduce_sum(y_true, axis=axis) + tf.reduce_sum(y_pred, axis=axis) + smooth)
    return tf.reduce_mean(batch_dsc, axis=0)


def hard_dice_coef(y_true, y_pred):
    smooth = 1e-5
    y_true_h = tf.round(y_true)
    y_pred_h = tf.round(y_pred)
    intersection = tf.reduce_sum(y_true_h * y_pred_h)
    batch_coef = (2. * intersection + smooth) / (tf.reduce_sum(y_true_h) + tf.reduce_sum(y_pred_h) + smooth)
    return batch_coef


def hard_dice_coef_channelwise(y_true, y_pred):
    smooth = 1e-5
    axis = (1, 2)
    y_true_h = tf.round(y_true)
    y_pred_h = tf.round(y_pred)
    intersection = tf.reduce_sum(y_true_h * y_pred_h, axis=axis)
    batch_coef = (2. * intersection + smooth) / (
                tf.reduce_sum(y_true_h, axis=axis) + tf.reduce_sum(y_pred_h, axis=axis) + smooth)
    return tf.reduce_mean(batch_coef, axis=0)


def overall_iou(y_true, y_pred):
    threshold = 0.5
    smooth = 1e-5

    y_pred_thresh = tf.cast(y_pred > threshold, dtype=tf.float32)
    y_true_thresh = tf.cast(y_true > threshold, dtype=tf.float32)
    intersection = tf.reduce_sum(tf.multiply(y_pred_thresh, y_true_thresh))
    union = tf.reduce_sum(tf.cast(tf.add(y_pred_thresh, y_true_thresh) >= 1, dtype=tf.float32))

    batch_iou = (intersection + smooth) / (union + smooth)
    batch_iou = tf.reduce_mean(batch_iou)
    return batch_iou


def iou_channelwise(y_true, y_pred):
    threshold = 0.5
    axis = (1, 2)
    smooth = 1e-5

    y_pred_tresh = tf.cast(y_pred > threshold, dtype=tf.float32)
    y_true_thresh = tf.cast(y_true > threshold, dtype=tf.float32)
    intersection = tf.reduce_sum(tf.multiply(y_pred_tresh, y_true_thresh), axis=axis)
    union = tf.reduce_sum(tf.cast(tf.add(y_pred_tresh, y_true_thresh) >= 1, dtype=tf.float32), axis=axis)

    batch_iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(batch_iou, axis=0)


def mean_class_iou(y_true, y_pred):
    iou_coef = iou_channelwise(y_true, y_pred)[1:]
    return tf.reduce_mean(iou_coef)


def particle_iou(y_true, y_pred):
    return iou_channelwise(y_true, y_pred)[1]


def true_positive(y_true, y_pred):
    y_true = tf.reshape(tf.argmax(y_true, -1), [-1, 1])
    y_pred = tf.reshape(tf.argmax(y_pred, -1), [-1, 1])

    tp = tf.count_nonzero(y_true * y_pred)
    return tf.cast(tp, dtype=tf.float32)


def true_negative(y_true, y_pred):
    y_true = tf.reshape(tf.argmax(y_true, -1), [-1, 1])
    y_pred = tf.reshape(tf.argmax(y_pred, -1), [-1, 1])

    tn = tf.count_nonzero((y_true - 1) * (y_pred - 1))
    return tf.cast(tn, dtype=tf.float32)


def false_positive(y_true, y_pred):
    y_true = tf.reshape(tf.argmax(y_true, -1), [-1, 1])
    y_pred = tf.reshape(tf.argmax(y_pred, -1), [-1, 1])

    fp = tf.count_nonzero((y_true - 1) * y_pred)
    return tf.cast(fp, dtype=tf.float32)


def false_negative(y_true, y_pred):
    y_true = tf.reshape(tf.argmax(y_true, -1), [-1, 1])
    y_pred = tf.reshape(tf.argmax(y_pred, -1), [-1, 1])

    fn = tf.count_nonzero(y_true * (y_pred - 1))
    return tf.cast(fn, dtype=tf.float32)


def precision(y_true, y_pred):
    smooth = tf.constant([0.00001], dtype=tf.float32)
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    pr = (tp + smooth) / (tp + fp + smooth)

    return pr


def recall(y_true, y_pred):
    smooth = tf.constant([0.00001], dtype=tf.float32)
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    rc = (tp + smooth) / (tp + fn + smooth)

    return rc


def f1(y_true, y_pred):
    pr = precision(y_true, y_pred)
    rc = recall(y_true, y_pred)
    f1_ = (2 * pr * rc) / (pr + rc)

    return f1_
