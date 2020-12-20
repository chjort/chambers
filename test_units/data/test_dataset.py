import os

import pytest
import tensorflow as tf

from chambers.data.dataset import InterleaveImageDataset, _block_iter
from chambers.data.dataset import (
    _shuffle_repeat,
    _get_input_len,
    _random_upsample,
    set_n_parallel,
)
from chambers.data.read import read_nested_set, read_img_files


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


class TestDataset(tf.test.TestCase):
    nested_data_path = "test_units/sample_data/mnist/train"
    class_dirs = sorted(read_nested_set(nested_data_path))
    labels = list(range(len(class_dirs)))

    def test_set_n_parallel0(self):
        td = InterleaveImageDataset(
            class_dirs=self.class_dirs,
            labels=self.labels,
            class_cycle_length=5,
            images_per_block=2,
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

        td = InterleaveImageDataset(
            class_dirs=self.class_dirs,
            labels=self.labels,
            class_cycle_length=5,
            images_per_block=2,
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
        files = read_img_files(self.class_dir)

        block = _block_iter(
            files=files,
            label=self.label,
            block_length=2,
            block_bound=False,
            sample_block_random=False,
            seed=None,
        )

        labels = [self.label] * len(files)
        files_list = list(zip(files, labels))
        block_list = list(block)
        self.assertEqual(block_list, files_list)

    def test_block_iter1(self):
        files = read_img_files(self.class_dir)

        block_len = 2

        block = _block_iter(
            files=files,
            label=self.label,
            block_length=block_len,
            block_bound=True,
            sample_block_random=False,
            seed=None,
        )

        labels = [self.label] * len(files)
        files_list = list(zip(files, labels))[:block_len]
        block_list = list(block)
        self.assertEqual(block_list, files_list)

    def test_block_iter2(self):
        files = read_img_files(self.class_dir)

        block = _block_iter(
            files=files,
            label=self.label,
            block_length=2,
            block_bound=False,
            sample_block_random=True,
            seed=None,
        )

        labels = [self.label] * len(files)
        files_list = list(zip(files, labels))
        block_list = list(block)
        self.assertNotEqual(block_list, files_list)

    def test_block_iter3(self):
        files = read_img_files(self.class_dir)

        block_len = 2

        block = _block_iter(
            files=files,
            label=self.label,
            block_length=block_len,
            block_bound=True,
            sample_block_random=True,
            seed=None,
        )

        labels = [self.label] * len(files)
        files_list = list(zip(files, labels))[:block_len]
        block_list = list(block)
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
