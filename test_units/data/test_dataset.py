import os

import pytest
import tensorflow as tf

from chambers.augmentations.single_image_augmentations import resize
from chambers.data.dataset import InterleaveImageClassDataset, _block_iter, InterleaveImageTripletDataset, \
    InterleaveImageClassTripletDataset
from chambers.data.dataset import (
    _shuffle_repeat,
    _get_input_len,
    _random_upsample,
    set_n_parallel,
)
from chambers.data.io import match_nested_set, match_img_files


def _get_dataset_labels(dataset, is_batched=False):
    if is_batched:
        labels = [y for xb, yb in dataset.as_numpy_iterator() for y in yb]
    else:
        labels = [y for x, y in dataset.as_numpy_iterator()]
    return labels


class TestGetInputLen:
    def test_get_input_len0(self):
        inputs = ("a", "b")
        input_len = _get_input_len(inputs)
        assert input_len == 2

    def test_get_input_len1(self):
        inputs = (["a", "b", "c"], [1, 2, 3])
        input_len = _get_input_len(inputs)
        assert input_len == 3

    def test_get_input_len2(self):
        with pytest.raises(ValueError):
            inputs = 5
            input_len = _get_input_len(inputs)


class TestImageClassDataset(tf.test.TestCase):
    data_path = "test_units/sample_data/mnist/train"
    class_dirs = sorted(match_nested_set(data_path))
    labels = list(range(len(class_dirs)))
    nc = 5
    nb = 2

    def test_set_n_parallel0(self):
        td = InterleaveImageClassDataset(
            class_dirs=self.class_dirs,
            labels=self.labels,
            class_cycle_length=self.nc,
            images_per_block=self.nb,
            image_channels=3,
            block_bound=True,
            sample_block_random=True,
            shuffle=True,
            reshuffle_iteration=False,
            buffer_size=1024,
            seed=42,
            repeats=None,
        )
        self.assertAllEqual(td._num_parallel_calls, -1)

    def test_set_n_parallel1(self):
        set_n_parallel(3)

        td = InterleaveImageClassDataset(
            class_dirs=self.class_dirs,
            labels=self.labels,
            class_cycle_length=self.nc,
            images_per_block=self.nb,
            image_channels=3,
            block_bound=True,
            sample_block_random=True,
            shuffle=True,
            reshuffle_iteration=False,
            buffer_size=1024,
            seed=42,
            repeats=None,
        )
        self.assertAllEqual(td._num_parallel_calls, 3)

    def test_block_bound0(self):
        td = InterleaveImageClassDataset(
            class_dirs=self.class_dirs,
            labels=self.labels,
            class_cycle_length=self.nc,
            images_per_block=self.nb,
            image_channels=3,
            block_bound=True,
            sample_block_random=False,
            shuffle=False,
            reshuffle_iteration=False,
            buffer_size=1024,
            seed=None,
            repeats=None,
        )
        element_labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]
        non_batched_labels = _get_dataset_labels(td, is_batched=False)
        batched_labels = _get_dataset_labels(
            td.batch(self.nc * self.nb), is_batched=True
        )
        self.assertEqual(non_batched_labels, element_labels)
        self.assertEqual(batched_labels, element_labels)

    def test_block_bound1(self):
        td = InterleaveImageClassDataset(
            class_dirs=self.class_dirs,
            labels=self.labels,
            class_cycle_length=self.nc,
            images_per_block=self.nb,
            image_channels=3,
            block_bound=False,
            sample_block_random=False,
            shuffle=False,
            reshuffle_iteration=False,
            buffer_size=1024,
            seed=None,
            repeats=None,
        )
        element_labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 5, 6, 7, 8, 9]
        non_batched_labels = _get_dataset_labels(td, is_batched=False)
        batched_labels = _get_dataset_labels(
            td.batch(self.nc * self.nb), is_batched=True
        )
        self.assertEqual(non_batched_labels, element_labels)
        self.assertEqual(batched_labels, element_labels)

    def test_random0(self):
        td = InterleaveImageClassDataset(
            class_dirs=self.class_dirs,
            labels=self.labels,
            class_cycle_length=self.nc,
            images_per_block=self.nb,
            image_channels=3,
            block_bound=True,
            sample_block_random=True,
            shuffle=True,
            reshuffle_iteration=False,
            buffer_size=1024,
            seed=42,
            repeats=None,
        )
        element_labels = [2, 2, 0, 0, 7, 7, 8, 8, 9, 9, 3, 3, 6, 6, 4, 4, 1, 1, 5, 5]
        non_batched_labels = _get_dataset_labels(td, is_batched=False)
        batched_labels = _get_dataset_labels(
            td.batch(self.nc * self.nb), is_batched=True
        )
        self.assertEqual(non_batched_labels, element_labels)
        self.assertEqual(batched_labels, element_labels)


class TestImageTripletDataset(tf.test.TestCase):
    data_path = "test_units/sample_data/triplets/train"
    class_dirs = sorted(match_nested_set(data_path))
    labels = list(range(len(class_dirs)))
    nc = 5
    nb = 2

    def test_block_bound0(self):
        td = InterleaveImageTripletDataset(
            class_dirs=self.class_dirs,
            labels=self.labels,
            class_cycle_length=self.nc,
            images_per_block=self.nb,
            image_channels=3,
            block_bound=True,
            sample_block_random=False,
            shuffle=False,
            reshuffle_iteration=False,
            buffer_size=1024,
            seed=None,
            repeats=None,
        )
        td = td.map(lambda x, y: (resize(x, (224, 224)), y))

        element_labels = [0, -1, 1, -1, 2, -1, 3, -1, 4, -1]
        non_batched_labels = _get_dataset_labels(td, is_batched=False)
        batched_labels = _get_dataset_labels(
            td.batch(self.nc * self.nb), is_batched=True
        )
        self.assertEqual(non_batched_labels, element_labels)
        self.assertEqual(batched_labels, element_labels)

    def test_block_bound1(self):
        td = InterleaveImageTripletDataset(
            class_dirs=self.class_dirs,
            labels=self.labels,
            class_cycle_length=self.nc,
            images_per_block=self.nb,
            image_channels=3,
            block_bound=False,
            sample_block_random=False,
            shuffle=False,
            reshuffle_iteration=False,
            buffer_size=1024,
            seed=None,
            repeats=None,
        )
        td = td.map(lambda x, y: (resize(x, (224, 224)), y))

        element_labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, -1, -1, 2, -1, 3, 3, 4, 4, -1, -1, -1, -1, 3, 3, -1, -1,
                          -1, -1, -1, -1, 3, 3, -1, -1, -1, -1, -1, -1, 3, 3, -1, -1, -1, -1, 3, 3, -1, -1, 3, 3, -1,
                          -1, 3, 3, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          -1, -1, -1]
        non_batched_labels = _get_dataset_labels(td, is_batched=False)
        batched_labels = _get_dataset_labels(
            td.batch(self.nc * self.nb), is_batched=True
        )
        self.assertEqual(non_batched_labels, element_labels)
        self.assertEqual(batched_labels, element_labels)

    def test_random0(self):
        td = InterleaveImageTripletDataset(
            class_dirs=self.class_dirs,
            labels=self.labels,
            class_cycle_length=self.nc,
            images_per_block=self.nb,
            image_channels=3,
            block_bound=True,
            sample_block_random=True,
            shuffle=True,
            reshuffle_iteration=False,
            buffer_size=1024,
            seed=42,
            repeats=None,
        )
        td = td.map(lambda x, y: (resize(x, (224, 224)), y))

        element_labels = [2, -1, 1, -1, 3, -1, 4, -1, 0, -1]
        non_batched_labels = _get_dataset_labels(td, is_batched=False)
        batched_labels = _get_dataset_labels(
            td.batch(self.nc * self.nb), is_batched=True
        )
        self.assertEqual(non_batched_labels, element_labels)
        self.assertEqual(batched_labels, element_labels)


class TestInterleaveImageClassTripletDataset(tf.test.TestCase):
    data_path = "test_units/sample_data/mnist/train"
    class_dirs = sorted(match_nested_set(data_path))

    triplet_data_path = "test_units/sample_data/triplets/train"
    triplet_class_dirs = sorted(match_nested_set(triplet_data_path))
    triplet_labels = list(range(len(triplet_class_dirs)))

    class_dirs.extend(triplet_class_dirs)
    labels = list(range(len(class_dirs)))

    nc = 5
    nb = 2

    def test_block_bound0(self):
        td = InterleaveImageClassTripletDataset(
            class_dirs=self.class_dirs,
            labels=self.labels,
            class_cycle_length=self.nc,
            images_per_block=self.nb,
            image_channels=3,
            block_bound=True,
            sample_block_random=False,
            shuffle=False,
            reshuffle_iteration=False,
            buffer_size=1024,
            seed=None,
            repeats=None,
        )
        td = td.map(lambda x, y: (resize(x, (224, 224)), y))

        element_labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, -1, 11, -1, 12, -1, 13, -1,
                          14, -1]
        non_batched_labels = _get_dataset_labels(td, is_batched=False)
        batched_labels = _get_dataset_labels(
            td.batch(self.nc * self.nb), is_batched=True
        )
        self.assertEqual(non_batched_labels, element_labels)
        self.assertEqual(batched_labels, element_labels)

    def test_block_bound1(self):
        td = InterleaveImageClassTripletDataset(
            class_dirs=self.class_dirs,
            labels=self.labels,
            class_cycle_length=self.nc,
            images_per_block=self.nb,
            image_channels=3,
            block_bound=False,
            sample_block_random=False,
            shuffle=False,
            reshuffle_iteration=False,
            buffer_size=1024,
            seed=None,
            repeats=None,
        )
        td = td.map(lambda x, y: (resize(x, (224, 224)), y))

        element_labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 5, 6, 7, 8, 9, 10,
                          10, 11, 11, 12, 12, 13, 13, 14, 14, 10, 10, -1, -1, 12, -1, 13, 13, 14, 14, -1, -1, -1, -1,
                          13, 13, -1, -1, -1, -1, -1, -1, 13, 13, -1, -1, -1, -1, -1, -1, 13, 13, -1, -1, -1, -1, 13,
                          13, -1, -1, 13, 13, -1, -1, 13, 13, -1, -1, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        non_batched_labels = _get_dataset_labels(td, is_batched=False)
        batched_labels = _get_dataset_labels(
            td.batch(self.nc * self.nb), is_batched=True
        )
        self.assertEqual(non_batched_labels, element_labels)
        self.assertEqual(batched_labels, element_labels)

    def test_random0(self):
        td = InterleaveImageClassTripletDataset(
            class_dirs=self.class_dirs,
            labels=self.labels,
            class_cycle_length=self.nc,
            images_per_block=self.nb,
            image_channels=3,
            block_bound=True,
            sample_block_random=True,
            shuffle=True,
            reshuffle_iteration=False,
            buffer_size=1024,
            seed=42,
            repeats=None,
        )
        td = td.map(lambda x, y: (resize(x, (224, 224)), y))

        element_labels = [2, 2, 1, 1, 5, 5, 4, 4, 9, 9, 13, -1, 3, 3, 10, -1, 7, 7, 0, 0, 11, -1, 12, -1, 8, 8, 6, 6,
                          14, -1]
        non_batched_labels = _get_dataset_labels(td, is_batched=False)
        batched_labels = _get_dataset_labels(
            td.batch(self.nc * self.nb), is_batched=True
        )
        self.assertEqual(non_batched_labels, element_labels)
        self.assertEqual(batched_labels, element_labels)


class TestBlockIter(tf.test.TestCase):
    nested_data_path = "test_units/sample_data/mnist/train"
    class_dir = os.path.join(nested_data_path, "0")
    label = 0
    slices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_random_upsample0(self):
        upsampled = _random_upsample(self.slices, 20)
        self.assertAllEqual(tf.shape(upsampled), (20,))

    def test_random_upsample1(self):
        upsampled = _random_upsample(self.slices, len(self.slices))
        self.assertAllEqual(upsampled, self.slices)

    def test_block_iter0(self):
        files = match_img_files(self.class_dir)

        block_iter = _block_iter(
            block=files,
            label=self.label,
            block_length=2,
            block_bound=False,
            sample_block_random=False,
            seed=None,
        )

        labels = [self.label] * len(files)
        files_list = list(zip(files, labels))
        block_list = list(block_iter)
        self.assertEqual(block_list, files_list)

    def test_block_iter1(self):
        files = match_img_files(self.class_dir)

        block_len = 2

        block_iter = _block_iter(
            block=files,
            label=self.label,
            block_length=block_len,
            block_bound=True,
            sample_block_random=False,
            seed=None,
        )

        labels = [self.label] * len(files)
        files_list = list(zip(files, labels))[:block_len]
        block_list = list(block_iter)
        self.assertEqual(block_list, files_list)

    def test_block_iter2(self):
        files = match_img_files(self.class_dir)

        block_iter = _block_iter(
            block=files,
            label=self.label,
            block_length=2,
            block_bound=False,
            sample_block_random=True,
            seed=None,
        )

        labels = [self.label] * len(files)
        files_list = list(zip(files, labels))
        block_list = list(block_iter)
        self.assertNotEqual(block_list, files_list)

    def test_block_iter3(self):
        files = match_img_files(self.class_dir)

        block_len = 2

        block_iter = _block_iter(
            block=files,
            label=self.label,
            block_length=block_len,
            block_bound=True,
            sample_block_random=True,
            seed=None,
        )

        labels = [self.label] * len(files)
        files_list = list(zip(files, labels))[:block_len]
        block_list = list(block_iter)
        self.assertNotEqual(block_list, files_list)


class TestShuffleRepeat(tf.test.TestCase):
    slices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_shuffle_repeat0(self):
        td = tf.data.Dataset.from_tensor_slices(self.slices)
        td = _shuffle_repeat(
            td,
            shuffle=False,
            buffer_size=None,
            reshuffle_iteration=False,
            seed=None,
            repeats=None,
        )
        td_out = list(td.as_numpy_iterator())
        assert td_out == self.slices

    def test_shuffle_repeat1(self):
        td = tf.data.Dataset.from_tensor_slices(self.slices)
        td = _shuffle_repeat(
            td,
            shuffle=False,
            buffer_size=None,
            reshuffle_iteration=False,
            seed=None,
            repeats=3,
        )
        td_out = list(td.as_numpy_iterator())
        assert len(td_out) == 3 * len(self.slices)

    def test_shuffle_repeat2(self):
        td = tf.data.Dataset.from_tensor_slices(self.slices)
        td = _shuffle_repeat(
            td,
            shuffle=True,
            buffer_size=len(self.slices),
            reshuffle_iteration=False,
            seed=None,
            repeats=None,
        )
        td_out = list(td.as_numpy_iterator())
        assert td_out != self.slices

    def test_shuffle_repeat3(self):
        td = tf.data.Dataset.from_tensor_slices(self.slices)
        td = _shuffle_repeat(
            td,
            shuffle=True,
            buffer_size=len(self.slices),
            reshuffle_iteration=False,
            seed=None,
            repeats=2,
        )
        td_out = list(td.as_numpy_iterator())
        it1 = td_out[: len(self.slices)]
        it2 = td_out[len(self.slices) :]

        assert it1 == it2

    def test_shuffle_repeat4(self):
        td = tf.data.Dataset.from_tensor_slices(self.slices)
        td = _shuffle_repeat(
            td,
            shuffle=True,
            buffer_size=len(self.slices),
            reshuffle_iteration=True,
            seed=None,
            repeats=2,
        )
        td_out = list(td.as_numpy_iterator())
        it1 = td_out[: len(self.slices)]
        it2 = td_out[len(self.slices) :]

        assert it1 != it2
