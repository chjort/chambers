from functools import partial

import numpy as np
import tensorflow as tf

from chambers.data.io import (
    match_img_files,
    read_and_decode_image,
    match_img_files_triplet,
)

__CONFIG = {"N_PARALLEL": -1}


def set_n_parallel(n):
    __CONFIG["N_PARALLEL"] = n


def _shuffle_repeat(
    dataset: tf.data.Dataset,
    shuffle=False,
    buffer_size=None,
    reshuffle_iteration=True,
    seed=None,
    repeats=None,
):
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=buffer_size,
            seed=seed,
            reshuffle_each_iteration=reshuffle_iteration,
        )

    if repeats is not None:
        if repeats == -1 or repeats > 0:
            dataset = dataset.repeat(repeats)
        else:
            raise ValueError("'repeats' must be greater than zero or equal to -1.")

    return dataset


def _get_input_len(inputs):
    input_ndims = np.ndim(inputs)
    if input_ndims == 1:
        input_len = len(inputs)
    elif input_ndims > 1:
        input_len = len(inputs[0])
    else:
        raise ValueError("Input with 0 dimensions has no length.")

    return input_len


def _sequential_dataset(
    inputs,
    shuffle=False,
    reshuffle_iteration=True,
    buffer_size=None,
    seed=None,
    repeats=None,
):
    if buffer_size is None:
        buffer_size = _get_input_len(inputs)

    td = tf.data.Dataset.from_tensor_slices(inputs)
    td = _shuffle_repeat(
        td,
        shuffle=shuffle,
        buffer_size=buffer_size,
        reshuffle_iteration=reshuffle_iteration,
        seed=seed,
        repeats=repeats,
    )
    return td


def _random_upsample(x, n, seed=None):
    n_x = tf.shape(x)[0]
    diff = n - n_x
    random_indices = tf.random.uniform(
        shape=[diff], minval=0, maxval=n_x, dtype=tf.int32, seed=seed
    )
    extra_samples = tf.gather(x, random_indices)
    x = tf.concat([x, extra_samples], axis=0)
    return x


def _block_iter(
    block, label, block_length, block_bound=True, sample_block_random=False, seed=None
):
    n_files = tf.shape(block)[0]

    block_length = tf.cast(block_length, n_files.dtype)
    if n_files < block_length:
        block = _random_upsample(block, block_length)

    if sample_block_random:
        block = tf.random.shuffle(block, seed=seed)

    n_files = tf.shape(block)[0]
    labels = tf.tile([label], [n_files])
    labels = tf.cast(labels, tf.int64)

    block = tf.data.Dataset.from_tensor_slices((block, labels))

    if block_bound:
        block_length = tf.cast(block_length, tf.int64)
        block = block.take(block_length)

    return block.repeat(1)


def _block_iter_triplet(
    triplets,
    label,
    block_length,
    block_bound=True,
    sample_block_random=False,
    seed=None,
):
    anch, pos, neg = triplets
    pos = tf.concat([anch, pos], axis=0)

    n_pos_block = tf.cast(tf.math.floor(block_length / 2), tf.int32)
    n_neg_block = tf.cast(tf.math.ceil(block_length / 2), tf.int32)

    block_iter_pos = _block_iter(
        pos,
        label,
        n_pos_block,
        block_bound=block_bound,
        sample_block_random=sample_block_random,
        seed=seed,
    )
    block_iter_neg = _block_iter(
        neg,
        -1,
        n_neg_block,
        block_bound=block_bound,
        sample_block_random=sample_block_random,
        seed=seed,
    )

    block_iter = block_iter_pos.concatenate(block_iter_neg)
    return block_iter.repeat(1)


def _interleave_fn_image_files(
    input_dir,
    label,
    block_length,
    block_bound=True,
    sample_block_random=False,
    seed=None,
):
    img_files = match_img_files(input_dir)
    block_iter = _block_iter(
        img_files,
        label,
        block_length=block_length,
        block_bound=block_bound,
        sample_block_random=sample_block_random,
        seed=seed,
    )
    return block_iter


def _interleave_fn_triplet_files(
    input_dir,
    label,
    block_length,
    block_bound=True,
    sample_block_random=False,
    seed=None,
):
    triplets = match_img_files_triplet(input_dir)
    block_iter = _block_iter_triplet(
        triplets,
        label,
        block_length=block_length,
        block_bound=block_bound,
        sample_block_random=sample_block_random,
        seed=seed,
    )
    return block_iter


@tf.function
def _interleave_fn_image_triplet_files(
    input_dir,
    label,
    block_length,
    block_bound=True,
    sample_block_random=False,
    seed=None,
):

    img_files = match_img_files(input_dir)

    # if no images found in folder, assume it is a triplet folder
    if tf.shape(img_files)[0] == 0:
        triplets = match_img_files_triplet(input_dir)
        block_iter = _block_iter_triplet(
            triplets,
            label,
            block_length=block_length,
            block_bound=block_bound,
            sample_block_random=sample_block_random,
            seed=seed,
        )
    else:
        block_iter = _block_iter(
            img_files,
            label,
            block_length=block_length,
            block_bound=block_bound,
            sample_block_random=sample_block_random,
            seed=seed,
        )

    return block_iter


def _interleave_dataset(
    inputs,
    interleave_fn,
    cycle_length,
    block_length,
    shuffle=False,
    reshuffle_iteration=True,
    buffer_size=None,
    seed=None,
    repeats=None,
):
    td = _sequential_dataset(
        inputs,
        shuffle=shuffle,
        reshuffle_iteration=reshuffle_iteration,
        buffer_size=buffer_size,
        seed=seed,
        repeats=repeats,
    )
    td = td.interleave(
        interleave_fn,
        cycle_length=cycle_length,
        block_length=block_length,
        num_parallel_calls=__CONFIG["N_PARALLEL"],
    )
    return td


def InterleaveImageClassDataset(
    class_dirs: list,
    labels: list,
    class_cycle_length: int,
    images_per_block: int,
    image_channels=3,
    block_bound=True,
    sample_block_random=False,
    shuffle=False,
    reshuffle_iteration=True,
    buffer_size=None,
    seed=None,
    repeats=None,
) -> tf.data.Dataset:
    """
    Constructs a tensorflow.data.Dataset which loads images by interleaving through class folders.
    """

    interleave_fn = partial(
        _interleave_fn_image_files,
        block_length=images_per_block,
        block_bound=block_bound,
        sample_block_random=sample_block_random,
        seed=seed,
    )
    td = _interleave_dataset(
        inputs=(class_dirs, labels),
        interleave_fn=interleave_fn,
        cycle_length=class_cycle_length,
        block_length=images_per_block,
        shuffle=shuffle,
        reshuffle_iteration=reshuffle_iteration,
        buffer_size=buffer_size,
        seed=seed,
        repeats=repeats,
    )
    td = td.map(
        lambda x, y: (read_and_decode_image(x, channels=image_channels), y),
        num_parallel_calls=__CONFIG["N_PARALLEL"],
    )
    return td


def InterleaveImageTripletDataset(
    class_dirs: list,
    labels: list,
    class_cycle_length: int,
    images_per_block: int,
    image_channels=3,
    block_bound=True,
    sample_block_random=False,
    shuffle=False,
    reshuffle_iteration=True,
    buffer_size=None,
    seed=None,
    repeats=None,
) -> tf.data.Dataset:
    """
    Constructs a tensorflow.data.Dataset which loads images by interleaving through triplet folders.
    """

    interleave_fn = partial(
        _interleave_fn_triplet_files,
        block_length=images_per_block,
        block_bound=block_bound,
        sample_block_random=sample_block_random,
        seed=seed,
    )
    td = _interleave_dataset(
        inputs=(class_dirs, labels),
        interleave_fn=interleave_fn,
        cycle_length=class_cycle_length,
        block_length=images_per_block,
        shuffle=shuffle,
        reshuffle_iteration=reshuffle_iteration,
        buffer_size=buffer_size,
        seed=seed,
        repeats=repeats,
    )
    td = td.map(
        lambda x, y: (read_and_decode_image(x, channels=image_channels), y),
        num_parallel_calls=__CONFIG["N_PARALLEL"],
    )
    return td


def InterleaveImageClassTripletDataset(
    class_dirs: list,
    labels: list,
    class_cycle_length: int,
    images_per_block: int,
    image_channels=3,
    block_bound=True,
    sample_block_random=False,
    shuffle=False,
    reshuffle_iteration=True,
    buffer_size=None,
    seed=None,
    repeats=None,
) -> tf.data.Dataset:
    """
    Constructs a tensorflow.data.Dataset which loads images by interleaving through class folders and triplet folders.
    """

    interleave_fn = partial(
        _interleave_fn_image_triplet_files,
        block_length=images_per_block,
        block_bound=block_bound,
        sample_block_random=sample_block_random,
        seed=seed,
    )
    td = _interleave_dataset(
        inputs=(class_dirs, labels),
        interleave_fn=interleave_fn,
        cycle_length=class_cycle_length,
        block_length=images_per_block,
        shuffle=shuffle,
        reshuffle_iteration=reshuffle_iteration,
        buffer_size=buffer_size,
        seed=seed,
        repeats=repeats,
    )
    td = td.map(
        lambda x, y: (read_and_decode_image(x, channels=image_channels), y),
        num_parallel_calls=__CONFIG["N_PARALLEL"],
    )
    return td


def SequentialImageDataset(
    class_dirs: list,
    labels: list,
    image_channels=3,
    shuffle=False,
    reshuffle_iteration=True,
    buffer_size=None,
    seed=None,
    repeats=None,
) -> tf.data.Dataset:
    """
    Constructs a tensorflow.data.Dataset which sequentially loads images from input folders.
    """

    td = _sequential_dataset(
        inputs=(class_dirs, labels),
        shuffle=shuffle,
        reshuffle_iteration=reshuffle_iteration,
        buffer_size=buffer_size,
        seed=seed,
        repeats=repeats,
    )

    def flat_map_fn(input_dir, label):
        files = match_img_files(input_dir)
        n_files = tf.shape(files)[0]
        y = tf.tile([label], [n_files])
        return tf.data.Dataset.from_tensor_slices((files, y))

    td = td.flat_map(flat_map_fn)

    td = td.map(
        lambda x, y: (read_and_decode_image(x, channels=image_channels), y),
        num_parallel_calls=__CONFIG["N_PARALLEL"],
    )
    return td
