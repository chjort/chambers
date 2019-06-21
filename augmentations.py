import tensorflow as tf

def get_augmentation(aug_name):
    aug_map = {"random_crop": random_crop,
               "random_color": random_color,
               "flip_horizontal": random_flip_horizontal,
               "flip_vertical": random_flip_vertical,
               "transpose": random_transpose,
               "rotate_90": random_rot90,
               "random_rotation": random_rotation,
               }
    return aug_map[aug_name]


def augmentations_to_function(augmentations: list):
    def aug_fn(img, mask):
        for aug, args in augmentations:
            if args is not None:
                img, mask = get_augmentation(aug)(img, mask, **args)
            else:
                img, mask = get_augmentation(aug)(img, mask)
        return img, mask
    return aug_fn


def random_crop(img, mask, crop_shape=(256, 256)):
    with tf.name_scope("random_crop"):
        height, width = crop_shape
        input_shape = tf.shape(img)
        xlim = tf.cast(input_shape[0] - width, tf.float32)
        ylim = tf.cast(input_shape[1] - height, tf.float32)
        xcoord = tf.cast(tf.random_uniform([1], minval=0, maxval=xlim, seed=42), tf.int32)
        ycoord = tf.cast(tf.random_uniform([1], minval=0, maxval=ylim, seed=42), tf.int32)

        img = tf.image.crop_to_bounding_box(img, xcoord[0], ycoord[0], width, height)
        mask = tf.image.crop_to_bounding_box(mask, xcoord[0], ycoord[0], width, height)

    return img, mask


def random_flip_horizontal(img, mask, prob=0.5):
    with tf.name_scope("random_fliph"):
        do_hflip = tf.random_uniform([], seed=42) < prob
        img = tf.cond(do_hflip, lambda: tf.image.flip_left_right(img), lambda: img)
        mask = tf.cond(do_hflip, lambda: tf.image.flip_left_right(mask), lambda: mask)

    return img, mask


def random_flip_vertical(img, mask, prob=0.5):
    with tf.name_scope("random_flipv"):
        do_vflip = tf.random_uniform([], seed=42) < prob
        img = tf.cond(do_vflip, lambda: tf.image.flip_up_down(img), lambda: img)
        mask = tf.cond(do_vflip, lambda: tf.image.flip_up_down(mask), lambda: mask)

    return img, mask


def random_rotation(img, mask, min_angle=-180, max_angle=180, prob=0.5):
    with tf.name_scope("random_rotation"):
        do_rot = tf.random_uniform([], seed=42) < prob
        angle = tf.random_uniform([1], minval=min_angle, maxval=max_angle, seed=42)
        img = tf.cond(do_rot, lambda: tf.contrib.image.rotate(img, angle), lambda: img)
        mask = tf.cond(do_rot, lambda: tf.contrib.image.rotate(mask, angle), lambda: mask)

    return img, mask


def random_rot90(img, mask, prob=0.5):
    with tf.name_scope("random_rot90"):
        do_rot90 = tf.random_uniform([], seed=42) < prob
        img = tf.cond(do_rot90, lambda: tf.image.rot90(img), lambda: img)
        mask = tf.cond(do_rot90, lambda: tf.image.rot90(mask), lambda: mask)

    return img, mask


def random_transpose(img, mask, prob=0.5):
    with tf.name_scope("random_transpose"):
        do_transpose = tf.random_uniform([], seed=42) < prob
        img = tf.cond(do_transpose, lambda: tf.image.transpose_image(img), lambda: img)
        mask = tf.cond(do_transpose, lambda: tf.image.transpose_image(mask), lambda: mask)

    return img, mask


def random_brightness(img, mask, max_delta=0.5, prob=0.5):
    with tf.name_scope("random_brightness"):
        do_brightness = tf.random_uniform([], seed=42) < prob
        img = tf.cond(do_brightness, lambda: tf.image.random_brightness(img, max_delta=max_delta, seed=42), lambda: img)

    return img, mask


def random_color(img, mask, prob=0.5):
    with tf.name_scope("random_color"):
        do_color = tf.random_uniform([], seed=42) < prob
        mult = tf.concat([[1], tf.random.uniform([2], 0, 1)], axis=0)
        mult = tf.random.shuffle(mult, seed=42)
        img = tf.cond(do_color, lambda: img * mult, lambda: img)

    return img, mask


def random_contrast(img, mask, min=0., max=1., prob=0.5):
    with tf.name_scope("random_contrast"):
        do_contrast = tf.random_uniform([], seed=42) < prob
        img = tf.cond(do_contrast, lambda: tf.image.random_contrast(img, lower=min, upper=max, seed=42), lambda: img)

    return img, mask
