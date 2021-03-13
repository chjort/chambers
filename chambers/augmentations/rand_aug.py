import math
from functools import partial

import tensorflow as tf
import tensorflow_addons as tfa


def wrap(image):
    """Returns 'image' with an extra channel set to all 1s."""
    shape = tf.shape(image)
    extended_channel = tf.ones([shape[0], shape[1], 1], image.dtype)
    extended = tf.concat([image, extended_channel], 2)
    return extended


def unwrap(image, replace_value):
    """Unwraps an image produced by wrap.

    Where there is a 0 in the last channel for every spatial position,
    the rest of the three channels in that spatial dimension are grayed
    (set to 128).  Operations like translate and shear on a wrapped
    Tensor will leave 0s in empty locations.  Some transformations look
    at the intensity of values to do preprocessing, and we want these
    empty pixels to assume the 'average' value, rather than pure black.


    Args:
      image: A 3D Image Tensor with 4 channels.
      replace_value: A one or three value 1D tensor to fill empty pixels.

    Returns:
      image: A 3D image Tensor with 3 channels.
    """
    image_shape = tf.shape(image)
    # Flatten the spatial dimensions.
    flattened_image = tf.reshape(image, [-1, image_shape[2]])

    # Find all pixels where the last channel is zero.
    alpha_channel = flattened_image[:, 3]

    replace_value = tf.concat([replace_value, tf.ones([1], image.dtype)], 0)

    # Where they are zero, fill them in with 'replace'.
    flattened_image = tf.where(
        tf.expand_dims(tf.equal(alpha_channel, 0), -1),
        tf.ones_like(flattened_image, dtype=image.dtype) * replace_value,
        flattened_image,
    )

    image = tf.reshape(flattened_image, image_shape)
    image = tf.slice(image, [0, 0, 0], [image_shape[0], image_shape[1], 3])
    return image


def with_wrapping(fn, replace_value):
    def wrapped_fn(image, *args, **kwargs):
        image = wrap(image)
        output = fn(image, *args, **kwargs)
        output = unwrap(output, replace_value)
        return output

    return wrapped_fn


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
    #
    # We need to clip and then cast.
    return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)


##### Transforms ####
def autocontrast(image):
    """Implements Autocontrast function from PIL using TF ops.

    Args:
      image: A 3D uint8 tensor.

    Returns:
      The image after it has had autocontrast applied to it and will be of type
      uint8.
    """

    def scale_channel(image):
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = tf.cast(tf.reduce_min(image), tf.float32)
        hi = tf.cast(tf.reduce_max(image), tf.float32)

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = tf.cast(im, tf.float32) * scale + offset
            im = tf.clip_by_value(im, 0.0, 255.0)
            return tf.cast(im, tf.uint8)

        result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
        return result

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image[:, :, 0])
    s2 = scale_channel(image[:, :, 1])
    s3 = scale_channel(image[:, :, 2])
    image = tf.stack([s1, s2, s3], 2)
    return image


def equalize(image):
    """Implements Equalize function from PIL using TF ops."""

    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = tf.cast(im[:, :, c], tf.int32)
        # Compute the histogram of the image channel.
        histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = tf.where(tf.not_equal(histo, 0))
        nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (tf.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = tf.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return tf.clip_by_value(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        result = tf.cond(
            tf.equal(step, 0), lambda: im, lambda: tf.gather(build_lut(histo, step), im)
        )

        return tf.cast(result, tf.uint8)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = tf.stack([s1, s2, s3], 2)
    return image


def invert(image):
    """Inverts the image pixels."""
    image = tf.convert_to_tensor(image)
    return 255 - image


# TODO: Existing keras layer?
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
    return tfa.image.rotate(image, radians)


def posterize(image, bits):
    """Equivalent of PIL Posterize."""
    shift = 8 - bits
    return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)


def solarize(image, threshold=128):
    # For each pixel in the image, select the pixel
    # if the value is less than the threshold.
    # Otherwise, subtract 255 from the pixel.
    return tf.where(image < threshold, image, 255 - image)


def solarize_add(image, addition=0, threshold=128):
    # For each pixel in the image less than threshold
    # we add 'addition' amount to it and then clip the
    # pixel value to be between 0 and 255. The value
    # of 'addition' is between -128 and 128.
    added_image = tf.cast(image, tf.int64) + addition
    added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
    return tf.where(image < threshold, added_image, image)


def color(image, factor):
    """Equivalent of PIL Color."""
    degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
    return blend(degenerate, image, factor)


# TODO: Existing keras layer?
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


# TODO: Existing keras layer?
def brightness(image, factor):
    """Equivalent of PIL Brightness."""
    degenerate = tf.zeros_like(image)
    return blend(degenerate, image, factor)


def sharpness(image, factor):
    """Implements Sharpness function from PIL using TF ops."""
    orig_image = image
    image = tf.cast(image, tf.float32)
    # Make image 4D for conv operation.
    image = tf.expand_dims(image, 0)
    # SMOOTH PIL Kernel.
    kernel = (
        tf.constant(
            [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32, shape=[3, 3, 1, 1]
        )
        / 13.0
    )
    # Tile across channel dimension.
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    strides = [1, 1, 1, 1]
    degenerate = tf.nn.depthwise_conv2d(
        input=image, filter=kernel, strides=strides, padding="VALID", dilations=[1, 1]
    )
    degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
    degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

    # For the borders of the resulting image, fill in the values of the
    # original image.
    mask = tf.ones_like(degenerate)
    padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
    padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
    result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

    # Blend the final result.
    return blend(result, orig_image, factor)


def shear_x(image, level):
    """Equivalent of PIL Shearing in X dimension."""
    # Shear parallel to x axis is a projective transform
    # with a matrix form of:
    # [1  level
    #  0  1].
    return tfa.image.transform(image, [1.0, level, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


# pass wrapped image
def shear_y(image, level):
    """Equivalent of PIL Shearing in Y dimension."""
    # Shear parallel to y axis is a projective transform
    # with a matrix form of:
    # [1  0
    #  level  1].
    return tfa.image.transform(image, [1.0, 0.0, 0.0, level, 1.0, 0.0, 0.0, 0.0])


# TODO: Existing keras layer?
# pass wrapped image
def translate_x(image, pixels):
    """Equivalent of PIL Translate in X dimension."""
    return tfa.image.translate(image, [-pixels, 0])


# TODO: Existing keras layer?
# pass wrapped image
def translate_y(image, pixels):
    """Equivalent of PIL Translate in Y dimension."""
    return tfa.image.translate(image, [0, -pixels])


def cutout(image, pad_size, replace_value=0):
    """Apply cutout (https://arxiv.org/abs/1708.04552) to image.

    This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
    a random location within `img`. The pixel values filled in will be of the
    value `replace`. The located where the mask will be applied is randomly
    chosen uniformly over the whole image.

    Args:
      image: An image Tensor of type uint8.
      pad_size: Specifies how big the zero mask that will be generated is that
        is applied to the image. The mask will be of size
        (2*pad_size x 2*pad_size).
      replace_value: What pixel value to fill in the image in the area that has
        the cutout mask applied to it.

    Returns:
      An image Tensor that is of type uint8.
    """
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    # Sample the center location in the image where the zero mask will be applied.
    cutout_center_height = tf.random.uniform(
        shape=[], minval=0, maxval=image_height, dtype=tf.int32
    )

    cutout_center_width = tf.random.uniform(
        shape=[], minval=0, maxval=image_width, dtype=tf.int32
    )

    lower_pad = tf.maximum(0, cutout_center_height - pad_size)
    upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
    left_pad = tf.maximum(0, cutout_center_width - pad_size)
    right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

    cutout_shape = [
        image_height - (lower_pad + upper_pad),
        image_width - (left_pad + right_pad),
    ]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(
        tf.zeros(cutout_shape, dtype=image.dtype), padding_dims, constant_values=1
    )
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, 3])
    image = tf.where(
        tf.equal(mask, 0), tf.ones_like(image, dtype=image.dtype) * replace_value, image
    )
    return image


#### Augment ####
def _randomly_negate_value(tensor):
    """With 50% prob turn the tensor negative."""
    should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
    final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
    return final_tensor


def get_transform(magnitude, transform_name):
    replace_value = [128] * 3
    cutout_const = 40
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
        "ShearX": {
            "level": _randomly_negate_value(shear_level),
        },
        "ShearY": {
            "level": _randomly_negate_value(shear_level),
        },
        "TranslateX": {
            "pixels": _randomly_negate_value(translate_pixels),
        },
        "TranslateY": {
            "pixels": _randomly_negate_value(translate_pixels),
        },
        "Cutout": {
            "pad_size": int(magnitude_ratio * cutout_const),
            "replace_value": replace_value,
        },
    }

    name_transform_map = {
        "AutoContrast": autocontrast,
        "Equalize": equalize,
        "Invert": invert,
        "Rotate": with_wrapping(rotate, replace_value),
        "Posterize": posterize,
        "Solarize": solarize,
        "SolarizeAdd": solarize_add,
        "Color": color,
        "Contrast": contrast,
        "Brightness": brightness,
        "Sharpness": sharpness,
        "ShearX": with_wrapping(shear_x, replace_value),
        "ShearY": with_wrapping(shear_y, replace_value),
        "TranslateX": with_wrapping(translate_x, replace_value),
        "TranslateY": with_wrapping(translate_y, replace_value),
        "Cutout": cutout,
    }

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


def rand_augment(image, n, magnitude):
    k = len(available_transforms)
    magnitude = float(magnitude)

    for i in range(n):
        transform_idx = tf.random.uniform([], maxval=k, dtype=tf.int32)
        for j, transform_name in enumerate(available_transforms):
            transform = get_transform(magnitude, transform_name)
            image = tf.cond(
                pred=tf.equal(j, transform_idx),
                true_fn=lambda: transform(image),
                false_fn=lambda: image,
            )
    return image
