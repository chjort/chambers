import tensorflow as tf
from chambers import metrics


def hard_dice_loss(y_true, y_pred):
    return 1 - metrics.hard_dice_coef(y_true, y_pred)


def soft_dice_loss(y_true, y_pred):
    return 1 - metrics.soft_dice_coef(y_true, y_pred)


def soft_dice_loss_no_bg(y_true, y_pred):
    dice_coefs = metrics.soft_dice_coef_channelwise(y_true, y_pred)[:, 1:]
    return tf.reduce_sum(1 - dice_coefs)


def cross_entropy_with_dice_coef(y_true, y_pred):
    cross_entropy_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    dice_loss = soft_dice_loss(y_true, y_pred)
    return tf.keras.backend.mean(cross_entropy_loss) + dice_loss


def cross_entropy_with_dice_coef_no_bg(y_true, y_pred):
    cross_entropy_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    dice_loss = soft_dice_loss_no_bg(y_true, y_pred)
    return tf.keras.backend.mean(cross_entropy_loss) + dice_loss
