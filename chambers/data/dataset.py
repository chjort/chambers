from functools import partial

import numpy as np
import tensorflow as tf

from chambers.data.read import read_img_files, read_and_decode_image

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
    files, label, block_length, block_bound=True, sample_block_random=False, seed=None
):
    n_files = tf.shape(files)[0]

    if n_files < block_length:
        files = _random_upsample(files, block_length)

    if sample_block_random:
        files = tf.random.shuffle(files, seed=seed)

    n_files = tf.shape(files)[0]
    labels = tf.tile([label], [n_files])
    labels = tf.cast(labels, tf.int64)

    block = tf.data.Dataset.from_tensor_slices((files, labels))

    if block_bound:
        block = block.take(block_length)
    return block


def _get_input_len(inputs):
    input_ndims = np.ndim(inputs)
    if input_ndims == 1:
        input_len = len(inputs)
    elif input_ndims > 1:
        input_len = len(inputs[0])
    else:
        raise ValueError("Input with 0 dimensions has no length.")

    return input_len


def _interleave_image_files(
    input_dir,
    label,
    block_length,
    block_bound=True,
    sample_block_random=False,
    seed=None,
):
    class_files = read_img_files(input_dir)
    block = _block_iter(
        class_files,
        label,
        block_length=block_length,
        block_bound=block_bound,
        sample_block_random=sample_block_random,
        seed=seed,
    )
    return block


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
    td = td.interleave(
        interleave_fn,
        cycle_length=cycle_length,
        block_length=block_length,
        num_parallel_calls=__CONFIG["N_PARALLEL"],
    )
    return td


def InterleaveImageDataset(
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
    interleave_fn = partial(
        _interleave_image_files,
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


def InterleaveTFRecordDataset(
    records: list,
    labels: list,
    record_cycle_length: int,
    examples_per_block: int,
    block_bound=True,
    sample_block_random=False,
    shuffle=False,
    reshuffle_iteration=True,
    buffer_size=None,
    seed=None,
    repeats=None,
) -> tf.data.Dataset:
    pass