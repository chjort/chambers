import tensorflow as tf


def random_rot90(x, seed=None):
    num_rots = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32, seed=seed)
    rot_x = tf.image.rot90(x, k=num_rots)
    return rot_x


def random_flip_up_down(x, seed=None):
    return tf.image.random_flip_up_down(x, seed=seed)


def random_flip_left_right(x, seed=None):
    return tf.image.random_flip_left_right(x, seed=seed)


def random_crop(x, height, width, seed=None):
    x_rank = x.shape.ndims

    if x_rank == 4:
        n = x.shape[0]
        channels = x.shape[3]
        cropped_x = tf.image.random_crop(x, [n, height, width, channels], seed=seed)
    elif x_rank == 3:
        channels = x.shape[2]
        cropped_x = tf.image.random_crop(x, [height, width, channels], seed=seed)
    else:
        raise ValueError("Input must have rank of 3 or 4. Found rank {}".format(x_rank))

    return cropped_x


def center_crop(x, height, width, input_height=None, input_width=None):
    x_rank = x.shape.ndims

    if input_height is not None:
        h = input_height
    elif x_rank == 4:
        h = x.shape[1]
    elif x_rank == 3:
        h = x.shape[0]
    else:
        raise ValueError("Input must have rank of 3 or 4. Found rank {}".format(x_rank))

    if input_width is not None:
        w = input_width
    elif x_rank == 4:
        w = x.shape[2]
    elif x_rank == 3:
        w = x.shape[1]
    else:
        raise ValueError("Input must have rank of 3 or 4. Found rank {}".format(x_rank))

    offset_h = tf.cast((h - height) / 2, tf.int32)
    offset_w = tf.cast((w - width) / 2, tf.int32)
    cropped_x = tf.image.crop_to_bounding_box(x, offset_height=offset_h, offset_width=offset_w,
                                              target_height=height, target_width=width)

    return cropped_x


def center_crop_fraction(x, center_fraction):
    return tf.image.central_crop(x, central_fraction=center_fraction)


def resize(x, height, width):
    """
    Resizes images to a specified height and width.

    :param x: numpy nd-array
    :param height: If float: scale the image height by h. If int resize image height to h.
    :param width: If float: scale the image width by w. If int resize image width to w.
    :return: numpy nd-array - the resized image.
    """

    # if h is a multiplier, multiply it with the original height
    if isinstance(height, float):
        imgh = tf.cast(tf.shape(x)[0], tf.float32)
        height = int(height * imgh)

    # if w is a multiplier, multiply it with the original width
    if isinstance(width, float):
        imgw = tf.cast(tf.shape(x)[1], tf.float32)
        width = int(width * imgw)

    x = tf.image.resize(x, (height, width))
    return x


def resize_max(x, max_size):
    h, w = x.shape[:2]
    if h > max_size:
        h = max_size / h
    if w > max_size:
        w = max_size / w

    x = resize(x, h, w)
    return x


def resnet_imagenet_normalize(x):
    x = tf.cast(x, tf.float32)

    # RGB -> BGR
    x = x[..., ::-1]
    x = normalize_image(x, mean=[103.939, 116.779, 123.68])

    return x


def torch_normalize(x):
    x = tf.cast(x, tf.float32)

    x = x / 255.
    x = normalize_image(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return x


def tf_normalize(x):
    x = tf.cast(x, tf.float32)

    x = x / 127.5
    x = x - 1.
    return x


def normalize_image(x, mean, std=None):
    """ Normalizes an image with mean and standard deviation """
    x = tf.cast(x, tf.float32)
    mean = tf.constant(mean, dtype=tf.float32)

    # subtract mean
    x_normed = (x - mean)

    if std is not None:
        # divide by standard deviation
        std = tf.constant(std, dtype=tf.float32)
        x_normed = x_normed / std

    return x_normed
