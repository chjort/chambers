from packaging import version
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.keras.engine.input_spec import InputSpec

from chambers.augmentations import image_augmentations

if version.parse(tf.__version__) < version.parse("2.4"):
    from tensorflow.python.keras.utils.tf_utils import smart_cond
else:
    from tensorflow.python.keras.utils.control_flow_util import smart_cond

_INTERPOLATION_MODE = "nearest"
_FILL_MODE = "constant"
_FILL_VALUE = 128
_MAX_MAGNITUDE = 10.0

_AUTO_AUGMENT_POLICY_V0 = [
    # [(Transform, Probability, Magnitude), (Transform, Probability, Magnitude)]
    [("Equalize", 0.8, None), ("ShearY", 0.8, 4)],
    [("Color", 0.4, 9), ("Equalize", 0.6, None)],
    [("Color", 0.4, 1), ("Rotate", 0.6, 8)],
    [("Solarize", 0.8, 3), ("Equalize", 0.4, 7)],
    [("Solarize", 0.4, 2), ("Solarize", 0.6, 2)],
    [("Color", 0.2, 0), ("Equalize", 0.8, None)],
    [("Equalize", 0.4, None), ("SolarizeAdd", 0.8, 3)],
    [("ShearX", 0.2, 9), ("Rotate", 0.6, 8)],
    [("Color", 0.6, 1), ("Equalize", 1.0, None)],
    [("Invert", 0.4, None), ("Rotate", 0.6, 0)],
    [("Equalize", 1.0, None), ("ShearY", 0.6, 3)],
    [("Color", 0.4, 7), ("Equalize", 0.6, None)],
    [("Posterize", 0.4, 6), ("AutoContrast", 0.4, None)],
    [("Solarize", 0.6, 8), ("Color", 0.6, 9)],
    [("Solarize", 0.2, 4), ("Rotate", 0.8, 9)],
    [("Rotate", 1.0, 7), ("TranslateY", 0.8, 9)],
    [("ShearX", 0.0, 0), ("Solarize", 0.8, 4)],
    [("ShearY", 0.8, 0), ("Color", 0.6, 4)],
    [("Color", 1.0, 0), ("Rotate", 0.6, 2)],
    [("Equalize", 0.8, None), ("Equalize", 0.0, None)],
    [("Equalize", 1.0, None), ("AutoContrast", 0.6, None)],
    [("ShearY", 0.4, 7), ("SolarizeAdd", 0.6, 7)],
    [("Posterize", 0.8, 2), ("Solarize", 0.6, 10)],
    [("Solarize", 0.6, 8), ("Equalize", 0.6, 1)],
    [("Color", 0.8, 6), ("Rotate", 0.4, 5)],
]


def _magnitude_to_enhance_kwargs(magnitude):
    factor = magnitude / _MAX_MAGNITUDE * 1.8 + 0.1
    kwargs = {"factor": factor}
    return kwargs


def _magnitude_to_shear_kwargs(magnitude):
    level = magnitude / _MAX_MAGNITUDE * 0.3
    kwargs = {
        "level": level,
        "interpolation": _INTERPOLATION_MODE,
        "fill_mode": _FILL_MODE,
        "fill_value": _FILL_VALUE,
    }
    return kwargs


def _magnitude_to_translate_kwargs(magnitude):
    pixels = magnitude / _MAX_MAGNITUDE * 100
    kwargs = {
        "pixels": pixels,
        "interpolation": _INTERPOLATION_MODE,
        "fill_mode": _FILL_MODE,
        "fill_value": _FILL_VALUE,
    }
    return kwargs


def _magnitude_to_posterize_kwargs(magnitude):
    bits = int(magnitude / _MAX_MAGNITUDE * 4)
    kwargs = {"bits": bits}
    return kwargs


def _magnitude_to_solarize_kwargs(magnitude):
    threshold = int(magnitude / _MAX_MAGNITUDE * 256)
    kwargs = {"threshold": threshold}
    return kwargs


def _magnitude_to_solarizeadd_kwargs(magnitude):
    addition = int(magnitude / _MAX_MAGNITUDE * 110)
    kwargs = {"addition": addition}
    return kwargs


def _magnitude_to_rotate_kwargs(magnitude):
    degrees = magnitude / _MAX_MAGNITUDE * 30.0
    kwargs = {
        "degrees": degrees,
        "interpolation": _INTERPOLATION_MODE,
        "fill_mode": _FILL_MODE,
        "fill_value": _FILL_VALUE,
    }
    return kwargs


def _magnitude_to_cutout_kwargs(magnitude):
    mask_size = int(magnitude / _MAX_MAGNITUDE * 80)
    kwargs = {"mask_size": mask_size, "constant_values": _FILL_VALUE}
    return kwargs


def _get_transform(transform_name, magnitude):
    magnitude_fn_map = {
        "AutoContrast": lambda magnitude: {},
        "Equalize": lambda magnitude: {},
        "Invert": lambda magnitude: {},
        "Brightness": _magnitude_to_enhance_kwargs,
        "Contrast": _magnitude_to_enhance_kwargs,
        "Color": _magnitude_to_enhance_kwargs,
        "Sharpness": _magnitude_to_enhance_kwargs,
        "ShearX": _magnitude_to_shear_kwargs,
        "ShearY": _magnitude_to_shear_kwargs,
        "TranslateX": _magnitude_to_translate_kwargs,
        "TranslateY": _magnitude_to_translate_kwargs,
        "Posterize": _magnitude_to_posterize_kwargs,
        "Solarize": _magnitude_to_solarize_kwargs,
        "SolarizeAdd": _magnitude_to_solarizeadd_kwargs,
        "CutOut": _magnitude_to_cutout_kwargs,
        "Rotate": _magnitude_to_rotate_kwargs,
    }

    transform = getattr(image_augmentations, transform_name)
    kwarg_fn = magnitude_fn_map[transform_name]
    kwargs = kwarg_fn(magnitude)
    return transform(**kwargs)


@tf.keras.utils.register_keras_serializable(package="Chambers")
class AutoAugment(preprocessing.PreprocessingLayer):
    """ Applies a random augmentation pair to each image """

    def __init__(self, elementwise=False, name=None, **kwargs):
        super(AutoAugment, self).__init__(name=name, **kwargs)
        self.elementwise = elementwise
        self.transforms = [
            tf.keras.Sequential(
                [
                    image_augmentations.RandomChance(_get_transform(t1, m1), p1),
                    image_augmentations.RandomChance(_get_transform(t2, m2), p2),
                ]
            )
            for (t1, p1, m1), (t2, p2, m2) in _AUTO_AUGMENT_POLICY_V0
        ]
        self._transform = image_augmentations.RandomChoice(
            self.transforms, n_transforms=1, elementwise=elementwise
        )
        self.input_spec = InputSpec(ndim=4, dtype=tf.uint8)

    def call(self, inputs, training=None, **kwargs):
        if training is None:
            training = tf.keras.backend.learning_phase()

        x = smart_cond(
            pred=training,
            true_fn=lambda: self._transform(inputs),
            false_fn=lambda: inputs,
        )
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "elementwise": self.elementwise,
        }
        base_config = super(AutoAugment, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="Chambers")
class RandAugment(preprocessing.PreprocessingLayer):
    def __init__(self, n_transforms, magnitude, elementwise=False, name=None, **kwargs):
        super(RandAugment, self).__init__(name=name, **kwargs)
        self.n_transforms = n_transforms
        self.magnitude = magnitude
        self.elementwise = elementwise
        self.transforms = [
            _get_transform("AutoContrast", magnitude),
            _get_transform("Equalize", magnitude),
            _get_transform("Invert", magnitude),
            _get_transform("Brightness", magnitude),
            _get_transform("Contrast", magnitude),
            _get_transform("Color", magnitude),
            _get_transform("Sharpness", magnitude),
            _get_transform("ShearX", magnitude),
            _get_transform("ShearY", magnitude),
            _get_transform("TranslateX", magnitude),
            _get_transform("TranslateY", magnitude),
            _get_transform("Posterize", magnitude),
            _get_transform("Solarize", magnitude),
            _get_transform("SolarizeAdd", magnitude),
            _get_transform("CutOut", magnitude),
            _get_transform("Rotate", magnitude),
        ]
        self._transform = image_augmentations.RandomChoice(
            self.transforms, n_transforms=n_transforms, elementwise=elementwise
        )
        self.input_spec = InputSpec(ndim=4, dtype=tf.uint8)

    def call(self, inputs, training=None, **kwargs):
        if training is None:
            training = tf.keras.backend.learning_phase()

        x = smart_cond(
            pred=training,
            true_fn=lambda: self._transform(inputs),
            false_fn=lambda: inputs,
        )
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "n_transforms": self.n_transforms,
            "magnitude": self.magnitude,
            "elementwise": self.elementwise,
        }
        base_config = super(RandAugment, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
