from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.losses import losses_utils
import metrics


class DSC(LossFunctionWrapper):
    def __init__(self,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='soft_dice_coefficient',
                 axis=None):
        super(DSC, self).__init__(
            soft_dice_loss, name=name, reduction=reduction, axis=axis)


class DHC(LossFunctionWrapper):
    def __init__(self,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='soft_dice_coefficient',
                 threshold=0.5,
                 axis=None):
        super(DHC, self).__init__(
            hard_dice_loss, name=name, reduction=reduction, threshold=threshold, axis=axis)


def hard_dice_loss(y_true, y_pred, threshold=0.5, axis=None):
    return 1 - metrics.hard_dice_coefficient(y_true, y_pred, threshold=threshold, axis=axis)


def soft_dice_loss(y_true, y_pred, axis=None):
    return 1 - metrics.soft_dice_coefficient(y_true, y_pred, axis=axis)

# def soft_dice_loss_no_bg(y_true, y_pred):
#     dice_coefs = metrics.soft_dice_coefficient(y_true, y_pred)[1:]
#
#     # weighting 2
#     n_classes = tf.keras.backend.int_shape(dice_coefs)[0]
#     scale = 0.4 / n_classes
#     dice_coefs = scale * dice_coefs
#
#     return tf.reduce_sum(1 - dice_coefs)
#
#
# def cross_entropy_with_dice_coef(y_true, y_pred):
#     cross_entropy_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
#     dice_loss = soft_dice_loss(y_true, y_pred)
#     return tf.keras.backend.mean(cross_entropy_loss) + dice_loss
#
#
# def cross_entropy_with_dice_coef_no_bg(y_true, y_pred):
#     cross_entropy_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
#     dice_loss = soft_dice_loss_no_bg(y_true, y_pred)
#
#     # weighting 1
#     #return tf.keras.backend.mean(cross_entropy_loss) + dice_loss
#
#     # weighting 2
#     return tf.keras.backend.mean(cross_entropy_loss) * 0.6 + dice_loss