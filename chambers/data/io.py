import glob
import os
from urllib.request import Request, urlopen

import tensorflow as tf

VALID_IMAGE_EXTENTIONS = [
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


def validate_dir_path(dir_path):
    """ Add "/" to dir_path if it does not already end with "/" """
    if tf.strings.substr(dir_path, -1, 1) != "/":
        dir_path = tf.strings.join([dir_path, "/"])
    return dir_path


def match_nested_set(path):
    return glob.glob(os.path.join(path, "*/"))


@tf.function
def match_img_files(dir_path):
    """
    Matches all .jpg, .png, .bmp and .gif files in a directory.

    :param dir_path: Path to directory containing files
    :return: 1-D Tensor containing file paths of matched files
    """
    dir_path = validate_dir_path(dir_path)

    # define patterns to match files with valid extensions
    patterns = []
    for ext in VALID_IMAGE_EXTENTIONS:
        pattern_glob = tf.strings.join([dir_path, "*.{}".format(ext)])
        patterns.append(pattern_glob)

    # get list of files matching the patterns
    files = tf.io.matching_files(pattern=patterns)

    return files


@tf.function
def match_img_files_triplet(dir_path):
    """
    Matches all .jpg, .png, .bmp and .gif files in a triplet directory.

    :param dir_path: Path to directory containing anchor, positive and negative subfolders
    :return: Tuple of 1-D Tensors containing file paths of achors, positives and negatives
    """
    dir_path = validate_dir_path(dir_path)

    anchor_files = match_img_files(tf.strings.join([dir_path, "anchor"]))
    positive_files = match_img_files(tf.strings.join([dir_path, "positive"]))
    negative_files = match_img_files(tf.strings.join([dir_path, "negative"]))
    return anchor_files, positive_files, negative_files


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


def open_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3"
    }
    req = Request(url, headers=headers)

    img_file = urlopen(req)
    return img_file


def read_url_bytes(url):
    img_file = open_url(url)
    return img_file.read()


def url_to_img(url, channels=3, expand_animations=False):
    img_bytes = read_url_bytes(url)
    img = tf.io.decode_image(
        img_bytes, channels=channels, expand_animations=expand_animations
    )
    return img
