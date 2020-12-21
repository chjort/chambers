import tensorflow as tf


def random_rot90(x, seed=None):
    num_rots = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32, seed=seed)
    rot_x = tf.image.rot90(x, k=num_rots)
    return rot_x


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
    cropped_x = tf.image.crop_to_bounding_box(
        x,
        offset_height=offset_h,
        offset_width=offset_w,
        target_height=height,
        target_width=width,
    )

    return cropped_x


def resize(x, size=None, min_side=None, max_side=None):
    if size is not None:
        h, w = size
        return _resize(x, h, w)
    else:
        return _resize_min_max(x, min_side, max_side)


def _resize(x, height, width):
    """
    Resizes images to a specified height and width.

    :param x: numpy nd-array
    :param height: If float: scale the image height by h. If int resize image height to h.
    :param width: If float: scale the image width by w. If int resize image width to w.
    :return: numpy nd-array - the resized image.
    """

    height = tf.convert_to_tensor(height)
    width = tf.convert_to_tensor(width)

    # if h is a multiplier, multiply it with the original height
    if height.dtype.is_floating:
        imgh = tf.cast(tf.shape(x)[0], tf.float32)
        height = tf.cast(height * imgh, tf.int32)

    # if w is a multiplier, multiply it with the original width
    if width.dtype.is_floating:
        imgw = tf.cast(tf.shape(x)[1], tf.float32)
        width = tf.cast(width * imgw, tf.int32)

    x = tf.image.resize(x, (height, width))
    return x


def _resize_min_max(x, min_side=None, max_side=None):
    """
    Resize an image to have its smallest side equal 'min_side' or its largest side equal 'max_side'.
    If both 'min_side' and 'max_side' is given, image will be resized to the side that scales down the image the most.
    Keeps aspect ratio of the image.
    """

    h = tf.cast(tf.shape(x)[0], tf.float32)
    w = tf.cast(tf.shape(x)[1], tf.float32)

    if min_side is not None and max_side is not None:
        cur_min_side = tf.minimum(w, h)
        min_side = tf.cast(min_side, tf.float32)
        cur_max_side = tf.maximum(w, h)
        max_side = tf.cast(max_side, tf.float32)
        scale = tf.minimum(max_side / cur_max_side, min_side / cur_min_side)
    elif min_side is not None:
        cur_min_side = tf.minimum(w, h)
        min_side = tf.cast(min_side, tf.float32)
        scale = min_side / cur_min_side
    elif max_side is not None:
        cur_max_side = tf.maximum(w, h)
        max_side = tf.cast(max_side, tf.float32)
        scale = max_side / cur_max_side
    else:
        raise ValueError("Must specify either 'min_side' or 'max_side'.")

    x = _resize(x, scale, scale)
    return x


def resnet_imagenet_normalize(x):
    x = tf.cast(x, tf.float32)

    # RGB -> BGR
    x = x[..., ::-1]
    x = normalize_image(x, mean=[103.939, 116.779, 123.68])

    return x


def torch_normalize(x):
    x = tf.cast(x, tf.float32)

    x = x / 255.0
    x = normalize_image(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return x


def tf_normalize(x):
    x = tf.cast(x, tf.float32)

    x = x / 127.5
    x = x - 1.0
    return x


def normalize_image(x, mean, std=None):
    """ Normalizes an image with mean and standard deviation """
    mean = tf.constant(mean, dtype=tf.float32)

    # subtract mean
    x_normed = x - mean

    if std is not None:
        # divide by standard deviation
        std = tf.constant(std, dtype=tf.float32)
        x_normed = x_normed / std

    return x_normed
