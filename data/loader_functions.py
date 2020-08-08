import tensorflow as tf


@tf.function
def match_img_files(dir_path):
    """
    Matches all .jpg, .png, .bmp and .gif files in a directory.

    :param dir_path: Path to directory containing files
    :return: 1-D Tensor containing file paths of matched files
    """
    if tf.strings.substr(dir_path, -1, 1) != "/":
        dir_path = tf.strings.join([dir_path, "/"])

    valid_extensions = ["jpg", "png", "bmp", "gif"]
    patterns = []
    for ext in valid_extensions:
        pattern_glob = tf.strings.join([dir_path, "*.{}".format(ext)])  # define pattern to match files with valid extensions
        patterns.append(pattern_glob)
    files = tf.io.matching_files(pattern=patterns)  # get list of files matching the patterns

    return files


def read_and_decode(file):
    """
    Read the bytes of an image file as RGB and decode the bytes into a tensor.
    Supports .png, .jpeg, .bmp and .gif files

    :param file: Path to the image file
    :return: RGB image as a 3-D tensor
    """
    channels = 3
    img_bytes = tf.io.read_file(file)
    img = tf.image.decode_image(img_bytes, channels=channels, expand_animations=False)
    img.set_shape([None, None, channels])

    return img