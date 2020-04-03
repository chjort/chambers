from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from ..augmentations import resize
from .loader_functions import match_img_files, read_and_decode
from .base_datasets import TensorDataset, TensorSliceDataset, GeneratorDataset

N_PARALLEL = tf.data.experimental.AUTOTUNE


class InterleaveDataset(TensorSliceDataset, ABC):
    """
        Constructs a tensorflow.data.Dataset which samples inputs by interleaving according to 'self.interleave_map'.
        For more detailed documentation see: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave

        The attribute 'self.dataset' is the tensorflow.data.Dataset producing outputs
    """

    def __init__(self, inputs, cycle_length=-1, block_length=1, shuffle=False, buffer_size=None, seed=None):
        """
            :param inputs: inputs to sample from
            :param cycle_length: number of inputs to iterate through before going to the next 'cycle_length' inputs.
            :param block_length: number of items to iterate through per input before moving to the next input in the
            cycle length.
            :param shuffle: If True the order of classes will be shuffled
            :param buffer_size: Size of the shuffle buffer
            :param seed: seed for the shuffle
            """

        input_ndims = np.ndim(inputs)
        if input_ndims == 1:
            input_len = len(inputs)
        elif input_ndims > 1:
            input_len = len(inputs[0])
            for dim in inputs:
                if len(dim) != input_len:
                    raise ValueError("All dimensions of input must have same length.")
        else:
            raise ValueError("Inputs must have rank higher than 0.")

        if cycle_length > input_len:
            raise ValueError(
                "Cycle length {} higher than input length {}. Cycle length must be lower than length of input.".format(
                    cycle_length, input_len)
            )

        super().__init__(inputs)
        self.cycle_length = cycle_length
        self.block_length = block_length
        self.do_shuffle = shuffle
        self.buffer_size = buffer_size
        self.seed = seed

        if self.buffer_size is None:
            self.buffer_size = input_len

        if self.do_shuffle:
            self.shuffle(buffer_size=self.buffer_size, seed=self.seed)

        self.interleave(self.interleave_fn, cycle_length=self.cycle_length,
                        block_length=self.block_length)

    @abstractmethod
    def interleave_fn(self, *args, **kwargs):
        pass


class LabeledImageDataset(InterleaveDataset):
    """
        Constructs a tensorflow.data.Dataset which generates batches of images and labels for a specified number of
        classes.

        The attribute 'self.dataset' is the tensorflow.data.Dataset producing outputs of (images, labels)
    """

    def __init__(self, image_class_dirs: list, labels: list, class_cycle_length, images_per_class_cycle,
                 sample_random=False, **kwargs):
        """
        :param image_class_dirs: list of class directories containing image files
        :param labels: list of labels for each class
        :param class_cycle_length: number of classes per cycle
        :param images_per_class_cycle: number of images per class in a cycle
        :param sample_random: Boolean. If true, will uniformly sample the images per class at random
        """
        self.sample_random = sample_random
        super().__init__((image_class_dirs, labels), cycle_length=class_cycle_length,
                         block_length=images_per_class_cycle,
                         **kwargs)

        self.map(read_and_decode)

    @tf.function
    def interleave_fn(self, input_dir, label):
        """
        TODO: Docstring

        :param input_dir:
        :param label:
        :return:
        """
        class_images = match_img_files(input_dir)
        n_files = tf.shape(class_images)[0]

        if n_files < self.block_length:
            # sample uniformly with replacement so there are 'block_length' number of files in the class
            diff = self.block_length - n_files
            random_indices = tf.random.uniform(shape=[diff], minval=0, maxval=n_files, dtype=tf.int32,
                                               seed=self.seed)
            extra_sampled_files = tf.gather(class_images, random_indices)
            class_images = tf.concat([class_images, extra_sampled_files], axis=0)
            n_files = tf.shape(class_images)[0]

        if self.sample_random:
            # sample 'block_length' samples uniformly without replacement
            uniform_dist = tf.random.uniform(shape=[n_files], minval=0, maxval=1, seed=self.seed)
            _, random_indices = tf.math.top_k(uniform_dist, k=self.block_length)
            class_images = tf.gather(class_images, random_indices)
            n_files = tf.shape(class_images)[0]

        labels = tf.tile([label], [n_files])
        return tf.data.Dataset.from_tensor_slices((class_images, labels))

    def map(self, func, *args, **kwargs):
        self.dataset = self.dataset.map(lambda images, labels: (func(images, *args, **kwargs), labels),
                                        num_parallel_calls=N_PARALLEL)


class RandomLabeledImageDataset(LabeledImageDataset):
    """
        Constructs a tensorflow.data.Dataset which generates batches of randomly sampled images and labels
        for a specified number of classes.

        The attribute 'self.dataset' is the tensorflow.data.Dataset producing outputs of (images, labels)
    """

    def __init__(self, image_class_dirs: list, labels: list, classes_per_batch, images_per_class,
                 resize_shape=(256, 256), seed=None):
        """
            :param image_class_dirs: list of class directories containing image files
            :param labels: list of labels for each class
            :param classes_per_batch: number of classes per batch
            :param images_per_class: number of images per class per batch
            :param resize_shape: tuple of height and width to resize images to
            :param seed: seed for the random sampling
        """
        super().__init__(image_class_dirs, labels, classes_per_batch, images_per_class,
                         sample_random=True, shuffle=True, buffer_size=None, seed=seed)

        if classes_per_batch < 1:
            raise ValueError("Classes per batch must be positive.")
        if images_per_class < 1:
            raise ValueError("Images per class must be positive.")

        self.map(resize, *resize_shape)

        # TODO: Dont batch here. Batch _after_ augmentations.
        self.batch(self.cycle_length * self.block_length, drop_remainder=True)


class EpisodeImageDataset(LabeledImageDataset):
    """
        Constructs a tensorflow.data.Dataset which generates batches of 'n-shot, k-way episodes'. Each batch consists of a
        dictionary containing a support set and a query set, and a one-hot encoded tensor as the labels
    """

    def __init__(self, image_class_dirs: list, labels: list, n, k, q, resize_shape=(224, 224)):
        """
        :param image_class_dirs:
        :param labels:
        :param n: Number of images per class in the support set
        :param k: Number of classes in the support set
        :param q: Number of images per class in the query set
        :param resize_shape: Tuple of (height, width) to resize image to
        """

        super().__init__(image_class_dirs, labels, class_cycle_length=k, images_per_class_cycle=n + q,
                         resize_shape=resize_shape, sample_random=True, shuffle=True)
        self.n = n
        self.k = k
        self.q = q
        self.dataset = self.dataset.batch(self.n + self.q, drop_remainder=True)
        self.dataset = self.dataset.batch(self.k, drop_remainder=True)
        self.dataset = self.dataset.map(self._get_support_query_y)

    @tf.function
    def _get_support_query_y(self, images, labels):
        """
        Returns a function that splits an interleave batch into a support set and a query set according
        to 'n', 'k' and 'q'.
        :return: Dictionary containing support set and query set and the one-hot encoded labels
        """

        S = images[:, :self.n, ...]
        Q = images[:, -self.q:, ...]
        Y = tf.tile(tf.reshape(tf.range(self.k), [-1, 1]), [1, self.q])
        Y = tf.one_hot(Y, self.k)

        return {"support": S, "query": Q}, Y
