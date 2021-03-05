import glob
import os

import tensorflow as tf


def match_nested_set(path):
    return glob.glob(os.path.join(path, "*/"))


@tf.function
def match_img_files(dir_path):
    """
    Matches all .jpg, .png, .bmp and .gif files in a directory.

    :param dir_path: Path to directory containing files
    :return: 1-D Tensor containing file paths of matched files
    """

    # add "/" to dir_path if it does not already end with "/"
    if tf.strings.substr(dir_path, -1, 1) != "/":
        dir_path = tf.strings.join([dir_path, "/"])

    valid_extensions = [
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "gif",
        "JPG",
        "JPEG",
        "PNG",
        "BMP",
        "GIF",
    ]
    patterns = []
    for ext in valid_extensions:
        pattern_glob = tf.strings.join(
            [dir_path, "*.{}".format(ext)]
        )  # define pattern to match files with valid extensions
        patterns.append(pattern_glob)
    files = tf.io.matching_files(
        pattern=patterns
    )  # get list of files matching the patterns

    return files


def read_and_decode_image(file, channels=3):
    """
    Read the bytes of an image file as RGB and decode the bytes into a tensor.
    Supports .png, .jpeg, .bmp and .gif files

    :param file: Path to the image file
    :return: RGB image as a 3-D tensor
    """
    img_bytes = tf.io.read_file(file)
    img = tf.image.decode_image(img_bytes, channels=channels, expand_animations=False)
    img.set_shape([None, None, channels])

    return img
