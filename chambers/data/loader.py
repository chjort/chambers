from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from einops import rearrange

from .base_datasets import TensorSliceDataset
from .loader_functions import match_img_files, read_and_decode
from .mixins import ImageLabelMixin
from .tf_record import batch_deserialize_tensor_example_uint8


class _TransformSliceDataset(TensorSliceDataset):
    def __init__(
        self,
        inputs,
        repeats=None,
        shuffle=False,
        buffer_size=None,
        reshuffle_iteration=True,
        seed=None,
    ):
        """
        :param inputs: inputs to sample from
        :param repeats: number of times to repeat dataset. Set repeats=-1 to repeat indefinitely.
        :param shuffle: If True the order of classes will be shuffled
        :param buffer_size: Size of the shuffle buffer
        :param reshuffle_iteration: If True, will reshuffle dataset each iteration. Otherwise shuffle only once.
        :param seed: seed for the shuffle
        """

        super().__init__(inputs)
        self.repeats = repeats
        self.do_shuffle = shuffle
        self.buffer_size = buffer_size
        self.reshuffle_iteration = reshuffle_iteration
        self.seed = seed

        if self.do_shuffle:

            if self.buffer_size is None:
                input_len = self._get_input_len(inputs)
                self.buffer_size = input_len

            self.shuffle(
                buffer_size=self.buffer_size,
                seed=seed,
                reshuffle_each_iteration=reshuffle_iteration,
            )

        if repeats is not None and repeats == -1:
            self.repeat()
        elif repeats is not None and repeats > 0:
            self.repeat(repeats)

    @staticmethod
    def _get_input_len(inputs):
        input_ndims = np.ndim(inputs)
        if input_ndims == 1:
            input_len = len(inputs)
        elif input_ndims > 1:
            input_len = len(inputs[0])
        else:
            raise ValueError("Input with 0 dimensions has no length.")

        return input_len


class InterleaveDataset(_TransformSliceDataset, ABC):
    """
    Constructs a tensorflow.data.Dataset which samples inputs by interleaving according to 'self.interleave_map'.
    For more detailed documentation see: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave

    The attribute 'self.dataset' is the tensorflow.data.Dataset producing outputs
    """

    def __init__(
        self,
        inputs,
        cycle_length=-1,
        block_length=1,
        repeats=None,
        shuffle=False,
        reshuffle_iteration=True,
        buffer_size=None,
        seed=None,
    ):
        """
        :param inputs: inputs to sample from
        :param cycle_length: number of inputs to iterate through before going to the next 'cycle_length' inputs.
        :param block_length: number of items to iterate through per input before moving to the next input in the
        cycle length.
        :param repeats: number of times to repeat dataset. Set repeats=-1 to repeat indefinitely.
        :param shuffle: If True the order of classes will be shuffled
        :param buffer_size: Size of the shuffle buffer
        :param seed: seed for the shuffle
        """

        super().__init__(
            inputs,
            repeats=repeats,
            shuffle=shuffle,
            reshuffle_iteration=reshuffle_iteration,
            buffer_size=buffer_size,
            seed=seed,
        )

        input_len = self._get_input_len(inputs)
        if cycle_length > input_len:
            raise ValueError(
                "Cycle length {} higher than input length {}. Cycle length must be lower than length of input.".format(
                    cycle_length, input_len
                )
            )

        self.cycle_length = cycle_length
        self.block_length = block_length

        self.interleave(
            self.interleave_fn,
            cycle_length=self.cycle_length,
            block_length=self.block_length,
        )

    @abstractmethod
    def interleave_fn(self, *args, **kwargs):
        pass


class InterleaveClassesDataset(InterleaveDataset):
    """
    Constructs a tensorflow.data.Dataset which loads images by interleaving through input folders.

    The attribute 'self.dataset' is the tensorflow.data.Dataset producing outputs of (images, labels)
    """

    def __init__(
        self,
        class_dirs: list,
        labels: list,
        class_cycle_length,
        n_per_class,
        block_bound=True,
        sample_n_random=False,
        repeats=None,
        shuffle=False,
        reshuffle_iteration=True,
        buffer_size=None,
        seed=None,
    ):
        """
        :param class_dirs: list of class directories
        :param labels: list of labels for each class
        :param class_cycle_length: number of classes per cycle
        :param n_per_class: number of elements per class
        :param sample_n_random: Boolean. If true, will uniformly sample the elements per class at random
        :param repeats: Number of times to iterate over the class dirs.
        :param shuffle: Shuffle the class dirs.
        :param reshuffle_iteration: If True and shuffle is True, will reshuffle the class dirs each iteration.
        :param buffer_size
        """
        self.sample_n_random = sample_n_random
        self.block_bound = block_bound
        super().__init__(
            (class_dirs, labels),
            cycle_length=class_cycle_length,
            block_length=n_per_class,
            repeats=repeats,
            shuffle=shuffle,
            reshuffle_iteration=reshuffle_iteration,
            buffer_size=buffer_size,
            seed=seed,
        )

    @abstractmethod
    def get_dir_files(self, input_dir):
        pass

    def random_upsample(self, x, n):
        n_x = tf.shape(x)[0]
        diff = n - n_x
        random_indices = tf.random.uniform(
            shape=[diff], minval=0, maxval=n_x, dtype=tf.int32, seed=self.seed
        )
        extra_samples = tf.gather(x, random_indices)
        x = tf.concat([x, extra_samples], axis=0)
        return x

    def block_iter(self, files, label):
        n_files = tf.shape(files)[0]

        if n_files < self.block_length:
            files = self.random_upsample(files, self.block_length)

        if self.sample_n_random:
            files = tf.random.shuffle(files, seed=self.seed)

        n_files = tf.shape(files)[0]
        labels = tf.tile([label], [n_files])

        block = tf.data.Dataset.from_tensor_slices((files, labels))

        if self.block_bound:
            block = block.take(self.block_length)  # TODO: Does this make a difference?
        return block

    @tf.function
    def interleave_fn(self, input_dir, label):
        class_files = self.get_dir_files(input_dir)
        return self.block_iter(class_files, label)


class InterleaveImageDataset(InterleaveClassesDataset, ImageLabelMixin):
    """
    Constructs a tensorflow.data.Dataset which loads images by interleaving through input folders.

    The attribute 'self.dataset' is the tensorflow.data.Dataset producing outputs of (images, labels)
    """

    def __init__(
        self,
        class_dirs: list,
        labels: list,
        class_cycle_length,
        n_per_class,
        block_bound=True,
        sample_n_random=False,
        repeats=None,
        shuffle=False,
        reshuffle_iteration=True,
        buffer_size=None,
        seed=None,
    ):
        """
        :param class_dirs: list of class directories
        :param labels: list of labels for each class
        :param class_cycle_length: number of classes per cycle
        :param n_per_class: number of elements per class
        :param sample_n_random: Boolean. If true, will uniformly sample the elements per class at random
        :param repeats: Number of times to iterate over the class dirs.
        :param shuffle: Shuffle the class dirs.
        :param reshuffle_iteration: If True and shuffle is True, will reshuffle the class dirs each iteration.
        :param buffer_size: Number of elements to fill the shuffle buffer with.
        :param seed: Seed to use for random shuffle and sampling.
        """
        super().__init__(
            class_dirs=class_dirs,
            labels=labels,
            class_cycle_length=class_cycle_length,
            n_per_class=n_per_class,
            block_bound=block_bound,
            sample_n_random=sample_n_random,
            repeats=repeats,
            shuffle=shuffle,
            reshuffle_iteration=reshuffle_iteration,
            buffer_size=buffer_size,
            seed=seed,
        )
        self.map_image(read_and_decode)

    def get_dir_files(self, input_dir):
        return match_img_files(input_dir)


class InterleaveTFRecordDataset(InterleaveDataset, ImageLabelMixin):
    """
    Constructs a tensorflow.data.Dataset which loads examples from TF Record files by interleaving through
    the input record files.
    """

    def __init__(
        self,
        records: list,
        record_cycle_length,
        n_per_record,
        block_bound=True,
        sample_n_random=False,
        repeats=None,
        shuffle=False,
        reshuffle_iteration=True,
        buffer_size=None,
        seed=None,
    ):
        """
        :param records: list of records
        :param record_cycle_length: number of records per cycle
        :param n_per_record: number of elements per record
        :param sample_n_random: Boolean. If true, will uniformly sample the elements per class at random
        :param repeats: Number of times to iterate over the class dirs.
        :param shuffle: Shuffle the class dirs.
        :param reshuffle_iteration: If True and shuffle is True, will reshuffle the class dirs each iteration.
        :param buffer_size: Number of elements to fill the shuffle buffer with.
        :param seed: Seed to use for random shuffle and sampling.
        """
        self.sample_n_random = sample_n_random
        self.block_bound = block_bound
        super().__init__(
            records,
            cycle_length=record_cycle_length,
            block_length=n_per_record,
            repeats=repeats,
            shuffle=shuffle,
            reshuffle_iteration=reshuffle_iteration,
            buffer_size=buffer_size,
            seed=seed,
        )

    @tf.function
    def interleave_fn(self, record):
        td_rec = tf.data.TFRecordDataset(record)
        if self.sample_n_random:
            td_rec = td_rec.shuffle(
                buffer_size=100, reshuffle_each_iteration=True, seed=self.seed
            )
            # td_rec = td_rec.repeat()
            if self.block_bound:
                td_rec = td_rec.take(self.block_length)
        return td_rec


class InterleaveOneshotDataset(InterleaveImageDataset):
    def __init__(
        self,
        class_dirs: list,
        labels: list,
        n: int,
        sample_n_random=True,
        repeats=None,
        shuffle=False,
        reshuffle_iteration=True,
        buffer_size=None,
        seed=None,
    ):
        """"""
        assert n >= 2 and n % 2 == 0, "n must be an even number and at least 2."
        super(InterleaveOneshotDataset, self).__init__(
            class_dirs=class_dirs,
            labels=labels,
            class_cycle_length=2,
            n_per_class=n,
            block_bound=False,
            sample_n_random=sample_n_random,
            repeats=repeats,
            shuffle=shuffle,
            reshuffle_iteration=reshuffle_iteration,
            buffer_size=buffer_size,
            seed=seed,
        )
        self.n = n
        self.batch(self.cycle_length * self.block_length, drop_remainder=True)
        self.map(self.arrange_oneshot)
        self.map(self.split_to_x1_x2_y)

    def map_images(self, func, *args, **kwargs):
        def fn(x1, x2, labels):
            return func(x1, *args, **kwargs), func(x2, *args, **kwargs), labels

        self.map(fn)

    def arrange_oneshot(self, x, y):
        pos = rearrange(
            x, "(n k) h w c -> n k h w c", k=self.cycle_length, n=self.block_length
        )
        neg = rearrange(
            x, "(k n) h w c -> n k h w c", k=self.cycle_length, n=self.block_length
        )
        x = tf.concat([pos, neg], axis=0)
        y = tf.concat([tf.ones(self.block_length), tf.zeros(self.block_length)], axis=0)
        y = tf.cast(y, tf.int32)
        return x, y

    def split_to_x1_x2_y(self, x, y):
        x1, x2 = tf.split(x, 2, axis=1)
        x1 = tf.squeeze(x1, 1)
        x2 = tf.squeeze(x2, 1)
        return (x1, x2), y


class InterleaveTFRecordOneshotDataset(InterleaveTFRecordDataset):
    def __init__(
        self,
        records: list,
        n: int,
        sample_n_random=True,
        repeats=None,
        shuffle=False,
        reshuffle_iteration=True,
        buffer_size=None,
        seed=None,
    ):
        """"""
        assert n >= 2 and n % 2 == 0, "n must be an even number and at least 2."
        super(InterleaveTFRecordOneshotDataset, self).__init__(
            records=records,
            record_cycle_length=2,
            n_per_record=n,
            block_bound=False,
            sample_n_random=sample_n_random,
            repeats=repeats,
            shuffle=shuffle,
            reshuffle_iteration=reshuffle_iteration,
            buffer_size=buffer_size,
            seed=seed,
        )
        self.n = n
        self.batch(self.cycle_length * self.block_length, drop_remainder=True)
        self.map(batch_deserialize_tensor_example_uint8)
        self.map(self.arrange_oneshot)
        self.map(self.split_to_x1_x2_y)

    def map_images(self, func, *args, **kwargs):
        def fn(x1, x2, labels):
            return func(x1, *args, **kwargs), func(x2, *args, **kwargs), labels

        self.map(fn)

    def arrange_oneshot(self, x, y):
        pos = rearrange(
            x, "(n k) h w c -> n k h w c", k=self.cycle_length, n=self.block_length
        )
        neg = rearrange(
            x, "(k n) h w c -> n k h w c", k=self.cycle_length, n=self.block_length
        )
        x = tf.concat([pos, neg], axis=0)
        y = tf.concat([tf.ones(self.block_length), tf.zeros(self.block_length)], axis=0)
        y = tf.cast(y, tf.int32)
        return x, y

    def split_to_x1_x2_y(self, x, y):
        x1, x2 = tf.split(x, 2, axis=1)
        x1 = tf.squeeze(x1, 1)
        x2 = tf.squeeze(x2, 1)
        return (x1, x2), y


class SequentialImageDataset(_TransformSliceDataset, ImageLabelMixin):
    """
    Constructs a tensorflow.data.Dataset which sequentially loads images from input folders.
    """

    def __init__(
        self,
        class_dirs: list,
        labels: list,
        repeats=None,
        shuffle=False,
        reshuffle_iteration=True,
        buffer_size=None,
        seed=None,
    ):
        """
        :param class_dirs: list of class directories containing image files
        :param labels: list of labels for each class
        :param class_cycle_length: number of classes per cycle
        :param images_per_class_cycle: number of images per class in a cycle
        :param sample_random: Boolean. If true, will uniformly sample the images per class at random
        """
        super().__init__(
            (class_dirs, labels),
            repeats=repeats,
            shuffle=shuffle,
            reshuffle_iteration=reshuffle_iteration,
            buffer_size=buffer_size,
            seed=seed,
        )

        self.flat_map(self.flat_map_fn)
        self.map_image(read_and_decode)

    @staticmethod
    def flat_map_fn(input_dir, label):
        files = match_img_files(input_dir)
        n_files = tf.shape(files)[0]
        y = tf.tile([label], [n_files])
        return tf.data.Dataset.from_tensor_slices((files, y))
