import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.keras.engine.input_spec import InputSpec
from chambers.augmentations.augment_schemes import (
    distort_image_with_autoaugment,
    distort_image_with_randaugment,
)
from chambers.augmentations.rand_aug import rand_augment


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


class AutoAugment(preprocessing.PreprocessingLayer):
    def __init__(self, name=None, **kwargs):
        super(AutoAugment, self).__init__(name=name, **kwargs)
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        fn = lambda x: distort_image_with_autoaugment(x, augmentation_name="v0")
        return tf.map_fn(fn, inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


class RandAugment(preprocessing.PreprocessingLayer):
    def __init__(self, num_layers, magnitude, name=None, **kwargs):
        super(RandAugment, self).__init__(name=name, **kwargs)
        self.num_layers = num_layers
        self.magnitude = magnitude
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        fn = lambda x: rand_augment(x, n=self.num_layers, magnitude=self.magnitude)
        return tf.map_fn(fn, inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
