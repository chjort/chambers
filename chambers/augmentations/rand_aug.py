import math

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.keras.engine.input_spec import InputSpec

_FILL_VALUE = 128
_INTERPOLATION_MODE = "nearest"


def blend(image1, image2, factor):
    """Blend image1 and image2 using 'factor'.

    Factor can be above 0.0.  A value of 0.0 means only image1 is used.
    A value of 1.0 means only image2 is used.  A value between 0.0 and
    1.0 means we linearly interpolate the pixel values between the two
    images.  A value greater than 1.0 "extrapolates" the difference
    between the two pixel values, and we clip the results to values
    between 0 and 255.

    Args:
      image1: An image Tensor of type uint8.
      image2: An image Tensor of type uint8.
      factor: A floating point value above 0.0.

    Returns:
      A blended image Tensor of type uint8.
    """
    if factor == 0.0:
        return tf.convert_to_tensor(image1)
    if factor == 1.0:
        return tf.convert_to_tensor(image2)

    image1 = tf.cast(image1, tf.float32)
    image2 = tf.cast(image2, tf.float32)

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = tf.cast(image1, tf.float32) + scaled

    # Interpolate
    if factor > 0.0 and factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        return tf.cast(temp, tf.uint8)

    # Extrapolate:
    # We need to clip and then cast.
    return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)


def _randomly_negate_value(value):
    """With 50% prob turn the value negative."""
    should_flip = tf.random.uniform([]) < 0.5
    value = tf.cond(should_flip, lambda: value, lambda: -value)
    return value


##### Transforms ####


class AutoContrast(preprocessing.PreprocessingLayer):
    def __init__(self, name=None, **kwargs):
        super(AutoContrast, self).__init__(name=name, **kwargs)
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        lo = tf.cast(tf.reduce_min(inputs, axis=(1, 2)), tf.float32)
        hi = tf.cast(tf.reduce_max(inputs, axis=(1, 2)), tf.float32)

        scale = tf.math.divide_no_nan(255.0, hi - lo)
        offset = -lo * scale

        # only scale channels where hi > lo
        mask = tf.cast(hi > lo, tf.float32)
        scale = scale * mask + (1 - mask)
        offset = offset * mask

        # expand dimensions so it is broadcastable to inputs
        scale = scale[:, None, None, :]
        offset = offset[:, None, None, :]

        x = tf.cast(inputs, tf.float32) * scale + offset
        x = tf.clip_by_value(x, 0.0, 255.0)
        x = tf.cast(x, tf.uint8)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class Equalize(preprocessing.PreprocessingLayer):
    def __init__(self, name=None, **kwargs):
        super(Equalize, self).__init__(name=name, **kwargs)
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        return tfa.image.equalize(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


class Invert(preprocessing.PreprocessingLayer):
    def __init__(self, name=None, **kwargs):
        super(Invert, self).__init__(name=name, **kwargs)
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        return 255 - inputs

    def compute_output_shape(self, input_shape):
        return input_shape


class Rotate(preprocessing.PreprocessingLayer):
    def __init__(self, degrees, name=None, **kwargs):
        super(Rotate, self).__init__(name=name, **kwargs)
        self.degrees = degrees
        self._radians = degrees * math.pi / 180.0
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        radians = _randomly_negate_value(self._radians)
        x = tfa.image.rotate(
            inputs,
            radians,
            interpolation=_INTERPOLATION_MODE,
            fill_mode="constant",
            fill_value=_FILL_VALUE,
        )
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"degrees": self.degrees}
        base_config = super(Rotate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Posterize(preprocessing.PreprocessingLayer):
    def __init__(self, bits, name=None, **kwargs):
        super(Posterize, self).__init__(name=name, **kwargs)
        self.bits = bits
        self._shift = 8 - bits
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        x = tf.bitwise.right_shift(inputs, self._shift)
        x = tf.bitwise.left_shift(x, self._shift)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"bits": self.bits}
        base_config = super(Posterize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Solarize(preprocessing.PreprocessingLayer):
    def __init__(self, threshold=128, name=None, **kwargs):
        super(Solarize, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        return tf.where(inputs < self.threshold, inputs, 255 - inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"threshold": self.threshold}
        base_config = super(Solarize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SolarizeAdd(preprocessing.PreprocessingLayer):
    def __init__(self, addition=0, threshold=128, name=None, **kwargs):
        super(SolarizeAdd, self).__init__(name=name, **kwargs)
        self.addition = addition
        self.threshold = threshold
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        x = tf.cast(inputs, tf.int64) + self.addition
        x = tf.cast(tf.clip_by_value(x, 0, 255), tf.uint8)
        return tf.where(inputs < self.threshold, x, inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"addition": self.addition, "threshold": self.threshold}
        base_config = super(SolarizeAdd, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Color(preprocessing.PreprocessingLayer):
    def __init__(self, factor, name=None, **kwargs):
        super(Color, self).__init__(name=name, **kwargs)
        self.factor = factor
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(inputs))
        return blend(degenerate, inputs, self.factor)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"factor": self.factor}
        base_config = super(Color, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Contrast(preprocessing.PreprocessingLayer):
    def __init__(self, factor, name=None, **kwargs):
        super(Contrast, self).__init__(name=name, **kwargs)
        self.factor = factor
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        degenerate = tf.image.rgb_to_grayscale(inputs)
        degenerate = tf.cast(degenerate, tf.int32)

        # Compute the grayscale histogram, then compute the mean pixel value,
        # and create a constant image size of that value.  Use that as the
        # blending degenerate target of the original image.
        hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
        mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
        degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
        degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
        degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
        return blend(degenerate, inputs, self.factor)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"factor": self.factor}
        base_config = super(Contrast, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Brightness(preprocessing.PreprocessingLayer):
    def __init__(self, factor, name=None, **kwargs):
        super(Brightness, self).__init__(name=name, **kwargs)
        self.factor = factor
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        degenerate = tf.zeros_like(inputs)
        return blend(degenerate, inputs, self.factor)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"factor": self.factor}
        base_config = super(Brightness, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Sharpness(preprocessing.PreprocessingLayer):
    def __init__(self, factor, name=None, **kwargs):
        super(Sharpness, self).__init__(name=name, **kwargs)
        self.factor = factor
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        return tfa.image.sharpness(inputs, self.factor)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"factor": self.factor}
        base_config = super(Sharpness, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ShearX(preprocessing.PreprocessingLayer):
    def __init__(self, level, name=None, **kwargs):
        super(ShearX, self).__init__(name=name, **kwargs)
        self.level = level
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        level = _randomly_negate_value(self.level)
        x = tfa.image.transform(
            inputs,
            [1.0, level, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            interpolation=_INTERPOLATION_MODE,
            fill_mode="constant",
            fill_value=_FILL_VALUE,
        )
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"level": self.level}
        base_config = super(ShearX, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ShearY(preprocessing.PreprocessingLayer):
    def __init__(self, level, name=None, **kwargs):
        super(ShearY, self).__init__(name=name, **kwargs)
        self.level = level
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        level = _randomly_negate_value(self.level)
        x = tfa.image.transform(
            inputs,
            [1.0, 0.0, 0.0, level, 1.0, 0.0, 0.0, 0.0],
            interpolation=_INTERPOLATION_MODE,
            fill_mode="constant",
            fill_value=_FILL_VALUE,
        )
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"level": self.level}
        base_config = super(ShearY, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TranslateX(preprocessing.PreprocessingLayer):
    def __init__(self, pixels, name=None, **kwargs):
        super(TranslateX, self).__init__(name=name, **kwargs)
        self.pixels = pixels
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        pixels = _randomly_negate_value(self.pixels)
        x = tfa.image.translate(
            inputs,
            [-pixels, 0],
            interpolation=_INTERPOLATION_MODE,
            fill_mode="constant",
            fill_value=_FILL_VALUE,
        )
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"pixels": self.pixels}
        base_config = super(TranslateX, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TranslateY(preprocessing.PreprocessingLayer):
    def __init__(self, pixels, name=None, **kwargs):
        super(TranslateY, self).__init__(name=name, **kwargs)
        self.pixels = pixels
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        pixels = _randomly_negate_value(self.pixels)
        x = tfa.image.translate(
            inputs,
            [0, -pixels],
            interpolation=_INTERPOLATION_MODE,
            fill_mode="constant",
            fill_value=_FILL_VALUE,
        )
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"pixels": self.pixels}
        base_config = super(TranslateY, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CutOut(preprocessing.PreprocessingLayer):
    def __init__(self, mask_size, name=None, **kwargs):
        super(CutOut, self).__init__(name=name, **kwargs)
        self.mask_size = mask_size
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        x = tfa.image.random_cutout(
            inputs, mask_size=self.mask_size, constant_values=_FILL_VALUE
        )
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "mask_size": self.mask_size,
        }
        base_config = super(CutOut, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
