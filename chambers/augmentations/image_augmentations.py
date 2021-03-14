import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.keras.engine.input_spec import InputSpec

from chambers.augmentations import rand_aug
from chambers.augmentations.augment_schemes import (
    distort_image_with_autoaugment,
)


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


policy = [
    # [(Transform, Probability, Magnitude), (Transform, Probability, Magnitude)]
    [(rand_aug.Equalize, 0.8, None), (rand_aug.ShearY, 0.8, 4)],
    [(rand_aug.Color, 0.4, 9), (rand_aug.Equalize, 0.6, None)],
    [(rand_aug.Color, 0.4, 1), (rand_aug.Rotate, 0.6, 8)],
    [(rand_aug.Solarize, 0.8, 3), (rand_aug.Equalize, 0.4, 7)],
    [(rand_aug.Solarize, 0.4, 2), (rand_aug.Solarize, 0.6, 2)],
    [(rand_aug.Color, 0.2, 0), (rand_aug.Equalize, 0.8, None)],
    [(rand_aug.Equalize, 0.4, None), (rand_aug.SolarizeAdd, 0.8, 3)],
    [(rand_aug.ShearX, 0.2, 9), (rand_aug.Rotate, 0.6, 8)],
    [(rand_aug.Color, 0.6, 1), (rand_aug.Equalize, 1.0, None)],
    [(rand_aug.Invert, 0.4, None), (rand_aug.Rotate, 0.6, 0)],
    [(rand_aug.Equalize, 1.0, None), (rand_aug.ShearY, 0.6, 3)],
    [(rand_aug.Color, 0.4, 7), (rand_aug.Equalize, 0.6, None)],
    [(rand_aug.Posterize, 0.4, 6), (rand_aug.AutoContrast, 0.4, None)],
    [(rand_aug.Solarize, 0.6, 8), (rand_aug.Color, 0.6, 9)],
    [(rand_aug.Solarize, 0.2, 4), (rand_aug.Rotate, 0.8, 9)],
    [(rand_aug.Rotate, 1.0, 7), (rand_aug.TranslateY, 0.8, 9)],
    [(rand_aug.ShearX, 0.0, 0), (rand_aug.Solarize, 0.8, 4)],
    [(rand_aug.ShearY, 0.8, 0), (rand_aug.Color, 0.6, 4)],
    [(rand_aug.Color, 1.0, 0), (rand_aug.Rotate, 0.6, 2)],
    [(rand_aug.Equalize, 0.8, None), (rand_aug.Equalize, 0.0, None)],
    [(rand_aug.Equalize, 1.0, None), (rand_aug.AutoContrast, 0.6, None)],
    [(rand_aug.ShearY, 0.4, 7), (rand_aug.SolarizeAdd, 0.6, 7)],
    [(rand_aug.Posterize, 0.8, 2), (rand_aug.Solarize, 0.6, 10)],
    [(rand_aug.Solarize, 0.6, 8), (rand_aug.Equalize, 0.6, 1)],
    [(rand_aug.Color, 0.8, 6), (rand_aug.Rotate, 0.4, 5)],
]


class AutoAugment2(preprocessing.PreprocessingLayer):
    """ Applies a random augmentation pair to each image """

    def __init__(self, name=None, **kwargs):
        super(AutoAugment2, self).__init__(name=name, **kwargs)

        self._max_magnitude = 10.0
        # self.transforms = [
        #     tf.keras.Sequential(
        #         [
        #             RandomChance(rand_aug.Equalize(), 0.8),
        #             RandomChance(
        #                 rand_aug.ShearY(level=self._shear_magnitude_to_level(4)), 0.8
        #             ),
        #         ]
        #     ),
        # ]

        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        fn = lambda x: distort_image_with_autoaugment(x, augmentation_name="v0")
        return tf.map_fn(fn, inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def _rotate_magnitude_to_degrees(self, magnitude):
        degrees = magnitude / self._max_magnitude * 30.0
        return degrees

    def _enhance_magnitude_to_factor(self, magnitude):
        return magnitude / self._max_magnitude * 1.8 + 0.1

    def _shear_magnitude_to_level(self, magnitude):
        magnitude = magnitude / self._max_magnitude * 0.3
        return magnitude

    def _translate_magnitude_to_pixels(self, magnitude):
        magnitude = magnitude / self._max_magnitude * 100
        return magnitude

    def _posterize_magnitude_to_bits(self, magnitude):
        magnitude = magnitude / self._max_magnitude * 4
        return magnitude

    def _solarize_magnitude_to_threshold(self, magnitude):
        magnitude = magnitude / self._max_magnitude * 256
        return magnitude

    def _solarizeadd_magnitude_to_addition(self, magnitude):
        magnitude = magnitude / self._max_magnitude * 110
        return magnitude


class RandAugment(preprocessing.PreprocessingLayer):
    def __init__(self, n_transforms, magnitude, separate=False, name=None, **kwargs):
        super(RandAugment, self).__init__(name=name, **kwargs)
        self.n_transforms = n_transforms
        self.magnitude = magnitude
        self.separate = separate

        max_magnitude = 10.0
        magnitude_ratio = magnitude / max_magnitude

        rotate_const = 30.0
        cutout_const = 80
        posterize_const = 4
        solarize_const = 256
        solarize_add_const = 110
        translate_const = 100

        enhance_factor = magnitude_ratio * 1.8 + 0.1
        shear_level = magnitude_ratio * 0.3
        translate_pixels = magnitude_ratio * translate_const

        self.transforms = [
            rand_aug.AutoContrast(),
            rand_aug.Equalize(),
            rand_aug.Invert(),
            rand_aug.Brightness(factor=enhance_factor),
            rand_aug.Contrast(factor=enhance_factor),
            rand_aug.Color(factor=enhance_factor),
            rand_aug.Sharpness(factor=enhance_factor),
            rand_aug.ShearX(level=shear_level),
            rand_aug.ShearY(level=shear_level),
            rand_aug.TranslateX(pixels=translate_pixels),
            rand_aug.TranslateY(pixels=translate_pixels),
            rand_aug.Posterize(bits=int(magnitude_ratio * posterize_const)),
            rand_aug.Solarize(threshold=int(magnitude_ratio * solarize_const)),
            rand_aug.SolarizeAdd(addition=int(magnitude_ratio * solarize_add_const)),
            rand_aug.CutOut(mask_size=int(magnitude_ratio * cutout_const)),
            rand_aug.Rotate(degrees=magnitude_ratio * rotate_const),
        ]

        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):

        if self.separate:
            inputs = tf.expand_dims(inputs, 1)
            x = tf.map_fn(self._rand_augment, inputs)
            x = tf.squeeze(x)
        else:
            x = self._rand_augment(inputs)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "n_transforms": self.n_transforms,
            "magnitude": self.magnitude,
            "separate": self.separate,
        }
        base_config = super(ResizingMinMax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _rand_augment(self, inputs):
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
