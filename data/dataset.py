import tensorflow as tf

from chambers.data.read import read_img_files


def _shuffle_repeat(
    dataset: tf.data.Dataset,
    shuffle=False,
    buffer_size=None,
    reshuffle_iteration=True,
    seed=None,
    repeats=None,
) -> tf.data.Dataset:
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
    files, label, block_length, block_bound=True, sample_n_random=False, seed=None
):
    n_files = tf.shape(files)[0]

    if n_files < block_length:
        files = _random_upsample(files, block_length)

    if sample_n_random:
        files = tf.random.shuffle(files, seed=seed)

    n_files = tf.shape(files)[0]
    labels = tf.tile([label], [n_files])

    block = tf.data.Dataset.from_tensor_slices((files, labels))

    if block_bound:
        block = block.take(block_length)
    return block


def interleave_image_files(
    input_dir, label, block_length, block_bound=True, sample_n_random=False, seed=None
):
    class_files = read_img_files(input_dir)
    block = _block_iter(
        class_files,
        label,
        block_length=block_length,
        block_bound=block_bound,
        sample_n_random=sample_n_random,
        seed=seed,
    )
    return block
