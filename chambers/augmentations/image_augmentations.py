import math
from typing import List

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.keras.engine.input_spec import InputSpec


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
    do_negate = tf.random.uniform([]) < 0.5
    value = tf.cond(do_negate, lambda: -value, lambda: value)
    return value


@tf.keras.utils.register_keras_serializable(package="Chambers")
class RandomChance(preprocessing.PreprocessingLayer):
    def __init__(
        self,
        transform: preprocessing.PreprocessingLayer,
        probability,
        name=None,
        **kwargs
    ):
        if name is None and transform.name is not None:
            name = "random_chance_" + transform.name
        super(RandomChance, self).__init__(name=name, **kwargs)
        self.transform = transform
        self.probability = probability

    def call(self, inputs, **kwargs):
        do_transform = tf.random.uniform([]) < self.probability
        x = tf.cond(
            pred=do_transform,
            true_fn=lambda: self.transform(inputs),
            false_fn=lambda: inputs,
        )
        return x

    def compute_output_shape(self, input_shape):
        return self.transform.compute_output_shape(input_shape)

    def get_config(self):
        config = {
            "transform": tf.keras.layers.serialize(self.transform),
            "probability": self.probability,
        }
        base_config = super(RandomChance, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        config["transform"] = tf.keras.layers.deserialize(config["transform"])
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="Chambers")
class RandomChoice(preprocessing.PreprocessingLayer):
    def __init__(
        self,
        transforms: List[preprocessing.PreprocessingLayer],
        n_transforms,
        elementwise=False,
        name=None,
        **kwargs
    ):
        super(RandomChoice, self).__init__(name=name, **kwargs)
        self.transforms = transforms
        self.n_transforms = n_transforms
        self.elementwise = elementwise

    def call(self, inputs, **kwargs):
        if self.elementwise:
            x = tf.expand_dims(inputs, 1)
            x = tf.map_fn(self._random_transforms, x)
            x = tf.squeeze(x)
        else:
            x = self._random_transforms(inputs)
        return x

    def compute_output_shape(self, input_shape):
        # check if all transforms has the same output shape
        output_shape0 = list(self.transforms[0].compute_output_shape(input_shape))
        is_equal = [
            list(transform.compute_output_shape(input_shape)) == output_shape0
            for transform in self.transforms
        ]
        if all(is_equal):
            return output_shape0
        else:
            # otherwise the output shape is unknown
            return [input_shape[0], None, None, None]

    def get_config(self):
        config = {
            "transforms": [
                tf.keras.layers.serialize(transform) for transform in self.transforms
            ],
            "n_transforms": self.n_transforms,
            "elementwise": self.elementwise,
        }
        base_config = super(RandomChoice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        config["transforms"] = [
            tf.keras.layers.deserialize(transform) for transform in config["transforms"]
        ]
        return cls(**config)

    def _random_transforms(self, inputs):
        for i in range(self.n_transforms):
            transform_idx = tf.random.uniform(
                [], maxval=len(self.transforms), dtype=tf.int32
            )
            for j, transform in enumerate(self.transforms):
                inputs = tf.cond(
                    pred=tf.equal(j, transform_idx),
                    true_fn=lambda: transform(inputs),
                    false_fn=lambda: inputs,
                )
        return inputs


@tf.keras.utils.register_keras_serializable(package="Chambers")
class ImageNetNormalization(preprocessing.PreprocessingLayer):
    def __init__(self, mode="caffe", name=None, **kwargs):
        super(ImageNetNormalization, self).__init__(name=name, **kwargs)
        if mode not in {"caffe", "tf", "torch"}:
            raise ValueError("Unknown mode " + str(mode))
        self.mode = mode
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        if self.mode == "tf":
            return self._tf_normalize(inputs)
        elif self.mode == "torch":
            return self._torch_normalize(inputs)
        else:
            return self._caffe_normalize(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "mode": self.mode,
        }
        base_config = super(ImageNetNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _caffe_normalize(self, x):
        x = x[..., ::-1]  # RGB -> BGR
        x = self._normalize(x, mean=[103.939, 116.779, 123.68])
        return x

    def _torch_normalize(self, x):
        x = tf.cast(x, tf.float32)

        x = x / 255.0
        x = self._normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return x

    @staticmethod
    def _tf_normalize(x):
        x = tf.cast(x, tf.float32)

        x = x / 127.5
        x = x - 1.0
        return x

    @staticmethod
    def _normalize(x, mean, std=None):
        """ Normalizes an image with mean and standard deviation """
        x = tf.cast(x, tf.float32)

        mean = tf.constant(mean, dtype=tf.float32)

        # subtract mean
        x_normed = x - mean

        if std is not None:
            # divide by standard deviation
            std = tf.constant(std, dtype=tf.float32)
            x_normed = x_normed / std

        return x_normed


@tf.keras.utils.register_keras_serializable(package="Chambers")
class ResizingMinMax(preprocessing.PreprocessingLayer):
    """
    Resize an image to have its smallest side equal 'min_side' or its largest side equal 'max_side'.
    If both 'min_side' and 'max_side' is given, image will be resized to the side that scales down the image the most.
    Keeps aspect ratio of the image.
    """

    def __init__(
        self,
        min_side=None,
        max_side=None,
        interpolation="bilinear",
        name=None,
        **kwargs
    ):
        super(ResizingMinMax, self).__init__(name=name, **kwargs)

        if min_side is None and max_side is None:
            raise ValueError("Must specify either 'min_side' or 'max_side'.")

        self.min_side = min_side
        self.max_side = max_side
        self.interpolation = interpolation
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        height = tf.cast(tf.shape(inputs)[1], tf.float32)
        width = tf.cast(tf.shape(inputs)[2], tf.float32)

        if self.min_side is not None and self.max_side is not None:
            cur_min_side = tf.minimum(width, height)
            min_side = tf.cast(self.min_side, tf.float32)
            cur_max_side = tf.maximum(width, height)
            max_side = tf.cast(self.max_side, tf.float32)
            scale = tf.minimum(max_side / cur_max_side, min_side / cur_min_side)
        elif self.min_side is not None:
            cur_min_side = tf.minimum(width, height)
            min_side = tf.cast(self.min_side, tf.float32)
            scale = min_side / cur_min_side
        else:
            cur_max_side = tf.maximum(width, height)
            max_side = tf.cast(self.max_side, tf.float32)
            scale = max_side / cur_max_side

        new_height = tf.cast(height * scale, tf.int32)
        new_width = tf.cast(width * scale, tf.int32)

        resized = preprocessing.Resizing(
            height=new_height, width=new_width, interpolation=self.interpolation
        )(inputs)
        return resized

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.min_side, self.max_side, input_shape[3]]

    def get_config(self):
        config = {
            "min_side": self.min_side,
            "max_side": self.max_side,
            "interpolation": self.interpolation,
        }
        base_config = super(ResizingMinMax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


####### Augmentations used by AutoAugment and RandAugment #######


@tf.keras.utils.register_keras_serializable(package="Chambers")
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


@tf.keras.utils.register_keras_serializable(package="Chambers")
class Equalize(preprocessing.PreprocessingLayer):
    def __init__(self, name=None, **kwargs):
        super(Equalize, self).__init__(name=name, **kwargs)
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        return tfa.image.equalize(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


@tf.keras.utils.register_keras_serializable(package="Chambers")
class Invert(preprocessing.PreprocessingLayer):
    def __init__(self, name=None, **kwargs):
        super(Invert, self).__init__(name=name, **kwargs)
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        return 255 - inputs

    def compute_output_shape(self, input_shape):
        return input_shape


@tf.keras.utils.register_keras_serializable(package="Chambers")
class Rotate(preprocessing.PreprocessingLayer):
    def __init__(
        self,
        degrees,
        interpolation="nearest",
        fill_mode="constant",
        fill_value=0.0,
        name=None,
        **kwargs
    ):
        super(Rotate, self).__init__(name=name, **kwargs)
        self.degrees = degrees
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self._radians = degrees * math.pi / 180.0
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        radians = _randomly_negate_value(self._radians)
        x = tfa.image.rotate(
            inputs,
            radians,
            interpolation=self.interpolation,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
        )
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "degrees": self.degrees,
            "interpolation": self.interpolation,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
        }
        base_config = super(Rotate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="Chambers")
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


@tf.keras.utils.register_keras_serializable(package="Chambers")
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


@tf.keras.utils.register_keras_serializable(package="Chambers")
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


@tf.keras.utils.register_keras_serializable(package="Chambers")
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


@tf.keras.utils.register_keras_serializable(package="Chambers")
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


@tf.keras.utils.register_keras_serializable(package="Chambers")
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


@tf.keras.utils.register_keras_serializable(package="Chambers")
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


@tf.keras.utils.register_keras_serializable(package="Chambers")
class ShearX(preprocessing.PreprocessingLayer):
    def __init__(
        self,
        level,
        interpolation="nearest",
        fill_mode="constant",
        fill_value=0.0,
        name=None,
        **kwargs
    ):
        super(ShearX, self).__init__(name=name, **kwargs)
        self.level = level
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        level = _randomly_negate_value(self.level)
        x = tfa.image.transform(
            inputs,
            [1.0, level, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            interpolation=self.interpolation,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
        )
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "level": self.level,
            "interpolation": self.interpolation,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
        }
        base_config = super(ShearX, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="Chambers")
class ShearY(preprocessing.PreprocessingLayer):
    def __init__(
        self,
        level,
        interpolation="nearest",
        fill_mode="constant",
        fill_value=0.0,
        name=None,
        **kwargs
    ):
        super(ShearY, self).__init__(name=name, **kwargs)
        self.level = level
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        level = _randomly_negate_value(self.level)
        x = tfa.image.transform(
            inputs,
            [1.0, 0.0, 0.0, level, 1.0, 0.0, 0.0, 0.0],
            interpolation=self.interpolation,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
        )
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "level": self.level,
            "interpolation": self.interpolation,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
        }
        base_config = super(ShearY, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="Chambers")
class TranslateX(preprocessing.PreprocessingLayer):
    def __init__(
        self,
        pixels,
        interpolation="nearest",
        fill_mode="constant",
        fill_value=0.0,
        name=None,
        **kwargs
    ):
        super(TranslateX, self).__init__(name=name, **kwargs)
        self.pixels = pixels
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        pixels = _randomly_negate_value(self.pixels)
        x = tfa.image.translate(
            inputs,
            [-pixels, 0],
            interpolation=self.interpolation,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
        )
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "pixels": self.pixels,
            "interpolation": self.interpolation,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
        }
        base_config = super(TranslateX, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="Chambers")
class TranslateY(preprocessing.PreprocessingLayer):
    def __init__(
        self,
        pixels,
        interpolation="nearest",
        fill_mode="constant",
        fill_value=0.0,
        name=None,
        **kwargs
    ):
        super(TranslateY, self).__init__(name=name, **kwargs)
        self.pixels = pixels
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        pixels = _randomly_negate_value(self.pixels)
        x = tfa.image.translate(
            inputs,
            [0, -pixels],
            interpolation=self.interpolation,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
        )
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "pixels": self.pixels,
            "interpolation": self.interpolation,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
        }
        base_config = super(TranslateY, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="Chambers")
class CutOut(preprocessing.PreprocessingLayer):
    def __init__(self, mask_size, constant_values=0, name=None, **kwargs):
        super(CutOut, self).__init__(name=name, **kwargs)
        self.mask_size = mask_size
        self.constant_values = constant_values
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        x = tfa.image.random_cutout(
            inputs, mask_size=self.mask_size, constant_values=self.constant_values
        )
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"mask_size": self.mask_size, "constant_values": self.constant_values}
        base_config = super(CutOut, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Grayscale(preprocessing.PreprocessingLayer):
    def __init__(self, keep_channels=False, **kwargs):
        super(Grayscale, self).__init__(**kwargs)
        self.keep_channels = keep_channels
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        x = tf.image.rgb_to_grayscale(inputs)
        if self.keep_channels:
            x = tf.concat([x, x, x], axis=-1)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"keep_channels": self.keep_channels}
        base_config = super(Grayscale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RGB(preprocessing.PreprocessingLayer):
    def __init__(self, **kwargs):
        super(RGB, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        x = tf.image.grayscale_to_rgb(inputs)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class GaussianBlur(preprocessing.PreprocessingLayer):
    def __init__(
        self, kernel_size, sigma=1.0, padding="CONSTANT", constant_values=0.0, **kwargs
    ):
        super(GaussianBlur, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = padding
        self.constant_values = constant_values

    def call(self, inputs, **kwargs):
        # TODO: Is this very slow?
        x = tfa.image.gaussian_filter2d(
            inputs,
            filter_shape=self.kernel_size,
            sigma=self.sigma,
            padding=self.padding,
            constant_values=self.constant_values,
        )
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "kernel_size": self.kernel_size,
            "sigma": self.sigma,
            "padding": self.padding,
            "constant_values": self.constant_values,
        }
        base_config = super(GaussianBlur, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
