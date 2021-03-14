import math
from functools import partial

import tensorflow as tf
import tensorflow_addons as tfa

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


##### Transforms ####
def autocontrast(images):
    rank = images.shape.ndims
    if rank == 4:
        axis = (1, 2)
    elif rank == 3:
        axis = (0, 1)
    else:
        raise ValueError(
            "Images must have rank 3 or rank 4. Found rank {}".format(rank)
        )

    lo = tf.cast(tf.reduce_min(images, axis=axis), tf.float32)
    hi = tf.cast(tf.reduce_max(images, axis=axis), tf.float32)

    scale = tf.math.divide_no_nan(255.0, hi - lo)
    offset = -lo * scale

    # only scale channels where hi > lo
    mask = tf.cast(hi > lo, tf.float32)
    scale = scale * mask + (1 - mask)
    offset = offset * mask

    # if tf.equal(rank, 4):
    if rank == 4:
        scale = scale[:, None, None, :]
        offset = offset[:, None, None, :]

    images = tf.cast(images, tf.float32) * scale + offset
    images = tf.clip_by_value(images, 0.0, 255.0)
    images = tf.cast(images, tf.uint8)
    return images


# NOTE: Layer
def equalize(image):
    return tfa.image.equalize(image)


# NOTE: Layer
def invert(image):
    """Inverts the image pixels."""
    image = tf.convert_to_tensor(image)
    return 255 - image


# NOTE: Layer
def rotate(image, degrees):
    """Rotates the image by degrees either clockwise or counterclockwise.

    Args:
      image: An image Tensor of type uint8.
      degrees: Float, a scalar angle in degrees to rotate all images by. If
        degrees is positive the image will be rotated clockwise otherwise it will
        be rotated counterclockwise.
      replace: A one or three value 1D tensor to fill empty pixels caused by
        the rotate operation.

    Returns:
      The rotated version of image.
    """
    # Convert from degrees to radians.
    degrees_to_radians = math.pi / 180.0
    radians = degrees * degrees_to_radians

    # In practice, we should randomize the rotation degrees by flipping
    # it negatively half the time, but that's done on 'degrees' outside
    # of the function.
    return tfa.image.rotate(
        image,
        radians,
        interpolation=_INTERPOLATION_MODE,
        fill_mode="constant",
        fill_value=_FILL_VALUE,
    )


# NOTE: Layer
def posterize(image, bits):
    """Equivalent of PIL Posterize."""
    shift = 8 - bits
    return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)


# NOTE: Layer
def solarize(image, threshold=128):
    # For each pixel in the image, select the pixel
    # if the value is less than the threshold.
    # Otherwise, subtract 255 from the pixel.
    return tf.where(image < threshold, image, 255 - image)


# NOTE: Layer
def solarize_add(image, addition=0, threshold=128):
    # For each pixel in the image less than threshold
    # we add 'addition' amount to it and then clip the
    # pixel value to be between 0 and 255. The value
    # of 'addition' is between -128 and 128.
    added_image = tf.cast(image, tf.int64) + addition
    added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
    return tf.where(image < threshold, added_image, image)


# NOTE: Layer
def color(image, factor):
    """Equivalent of PIL Color."""
    degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
    return blend(degenerate, image, factor)


# NOTE: Layer
def contrast(image, factor):
    """Equivalent of PIL Contrast."""
    degenerate = tf.image.rgb_to_grayscale(image)
    # Cast before calling tf.histogram.
    degenerate = tf.cast(degenerate, tf.int32)

    # Compute the grayscale histogram, then compute the mean pixel value,
    # and create a constant image size of that value.  Use that as the
    # blending degenerate target of the original image.
    hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
    mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
    degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
    degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
    degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
    return blend(degenerate, image, factor)


# NOTE: Layer
def brightness(image, factor):
    """Equivalent of PIL Brightness."""
    degenerate = tf.zeros_like(image)
    return blend(degenerate, image, factor)


# NOTE: Layer
def sharpness(image, factor):
    return tfa.image.sharpness(image, factor)


# NOTE: Layer
def shear_x(image, level):
    """Equivalent of PIL Shearing in X dimension."""
    return tfa.image.transform(
        image,
        [1.0, level, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        interpolation=_INTERPOLATION_MODE,
        fill_mode="constant",
        fill_value=_FILL_VALUE,
    )


# NOTE: Layer
def shear_y(image, level):
    """Equivalent of PIL Shearing in Y dimension."""
    return tfa.image.transform(
        image,
        [1.0, 0.0, 0.0, level, 1.0, 0.0, 0.0, 0.0],
        interpolation=_INTERPOLATION_MODE,
        fill_mode="constant",
        fill_value=_FILL_VALUE,
    )


# NOTE: Layer
def translate_x(image, pixels):
    """Equivalent of PIL Translate in X dimension."""
    return tfa.image.translate(
        image,
        [-pixels, 0],
        interpolation=_INTERPOLATION_MODE,
        fill_mode="constant",
        fill_value=_FILL_VALUE,
    )


# NOTE: Layer
def translate_y(image, pixels):
    """Equivalent of PIL Translate in Y dimension."""
    return tfa.image.translate(
        image,
        [0, -pixels],
        interpolation=_INTERPOLATION_MODE,
        fill_mode="constant",
        fill_value=_FILL_VALUE,
    )


# NOTE: Layer
def cutout(image, mask_size):
    rank = image.shape.ndims

    if rank == 3:
        image = tf.expand_dims(image, 0)

    image = tfa.image.random_cutout(
        image, mask_size=mask_size, constant_values=_FILL_VALUE
    )

    if rank == 3:
        image = image[0]

    return image


#### Augment ####
def _randomly_negate_value(tensor):
    """With 50% prob turn the tensor negative."""
    should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
    final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
    return final_tensor


def get_transform(magnitude, transform_name):
    cutout_const = 80
    translate_const = 100

    max_magnitude = 10.0
    magnitude_ratio = magnitude / max_magnitude

    enhance_factor = magnitude_ratio * 1.8 + 0.1
    shear_level = magnitude_ratio * 0.3
    translate_pixels = magnitude_ratio * translate_const

    name_kwargs_map = {
        "AutoContrast": {},
        "Equalize": {},
        "Invert": {},
        "Rotate": {
            "degrees": _randomly_negate_value(magnitude_ratio * 30.0),
        },
        "Posterize": {"bits": int(magnitude_ratio * 4)},
        "Solarize": {"threshold": int(magnitude_ratio * 256)},
        "SolarizeAdd": {"addition": int(magnitude_ratio * 110)},
        "Color": {"factor": enhance_factor},
        "Contrast": {"factor": enhance_factor},
        "Brightness": {"factor": enhance_factor},
        "Sharpness": {"factor": enhance_factor},
        "ShearX": {"level": _randomly_negate_value(shear_level)},
        "ShearY": {"level": _randomly_negate_value(shear_level)},
        "TranslateX": {"pixels": _randomly_negate_value(translate_pixels)},
        "TranslateY": {"pixels": _randomly_negate_value(translate_pixels)},
        "Cutout": {"mask_size": int(magnitude_ratio * cutout_const)},
    }

    name_transform_map = {
        "AutoContrast": autocontrast,
        "Equalize": equalize,
        "Invert": invert,
        "Rotate": rotate,
        # "Rotate": preprocessing.RandomRotation(
        #     factor=magnitude_ratio * 0.0833,
        #     fill_mode="constant",
        #     interpolation="nearest",
        #     fill_value=128,
        # ),
        "Posterize": posterize,
        "Solarize": solarize,
        "SolarizeAdd": solarize_add,
        "Color": color,
        "Contrast": contrast,
        "Brightness": brightness,
        "Sharpness": sharpness,
        "ShearX": shear_x,
        "ShearY": shear_y,
        "TranslateX": translate_x,
        "TranslateY": translate_y,
        "Cutout": cutout,
    }

    # if transform_name == "Rotate":
    #     def transform(image):
    #         rotate = name_transform_map[transform_name]
    #         image = tf.expand_dims(image, 0)
    #         return rotate(image)[0]
    #     return transform

    transform = partial(
        name_transform_map[transform_name],
        **name_kwargs_map[transform_name],
    )
    return transform


available_transforms = [
    "AutoContrast",
    "Equalize",
    "Invert",
    "Rotate",
    "Posterize",
    "Solarize",
    "SolarizeAdd",
    "Color",
    "Contrast",
    "Brightness",
    "Sharpness",
    "ShearX",
    "ShearY",
    "TranslateX",
    "TranslateY",
    "Cutout",
]


def rand_augment(image, n_transforms, magnitude):
    k = len(available_transforms)
    magnitude = float(magnitude)

    for i in range(n_transforms):
        transform_idx = tf.random.uniform([], maxval=k, dtype=tf.int32)
        for j, transform_name in enumerate(available_transforms):
            transform = get_transform(magnitude, transform_name)
            image = tf.cond(
                pred=tf.equal(j, transform_idx),
                true_fn=lambda: transform(image),
                false_fn=lambda: image,
            )
    return image
