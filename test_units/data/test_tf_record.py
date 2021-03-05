import pytest
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from chambers.data.dataset import InterleaveImageClassDataset
from chambers.data.load import match_nested_set
from chambers.data.tf_record import serialize_example, make_dataset_deserialize_fn


def random_size(x, y):
    size = tf.random.uniform((), minval=16, maxval=56, dtype=tf.int32)
    shape = (size, size)
    x = tf.image.resize(x, shape)
    return x, y


class TestTFRecord(tf.test.TestCase):
    nested_data_path = "test_units/sample_data/mnist/train"
    class_dirs = sorted(match_nested_set(nested_data_path))
    labels = list(range(len(class_dirs)))

    td = InterleaveImageClassDataset(
        class_dirs=class_dirs,
        labels=labels,
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

    def test_serialize_deserialize0(self):
        td = self.td
        batch = next(iter(td))

        td = td.map(serialize_example)
        td = td.map(make_dataset_deserialize_fn(td))

        batch_d = next(iter(td))
        self.assertAllEqual(batch[0], batch_d[0])
        self.assertAllEqual(batch[1], batch_d[1])

    def test_serialize_deserialize1(self):
        td = self.td
        td = td.map(lambda x, y: (x, tf.cast(x, tf.float32), y))
        batch = next(iter(td))

        td = td.map(serialize_example)
        td = td.map(make_dataset_deserialize_fn(td))

        batch_d = next(iter(td))

        self.assertAllEqual(batch[0], batch_d[0])
        self.assertAllEqual(batch[1], batch_d[1])
        self.assertAllEqual(batch[2], batch_d[2])

    def test_serialize_deserialize2(self):
        td = self.td
        td = td.map(lambda x, y: x)
        batch = next(iter(td))

        td = td.map(serialize_example)
        td = td.map(make_dataset_deserialize_fn(td))

        batch_d = next(iter(td))
        self.assertAllEqual(batch, batch_d)

    def test_serialize_deserialize_shape0(self):
        td = self.td
        batch = next(iter(td))

        td = td.map(serialize_example)
        td = td.map(make_dataset_deserialize_fn(td, set_shape=False, set_size=False))

        self.assertEqual(td.element_spec[0], tf.TensorSpec(shape=None, dtype=tf.uint8))
        self.assertEqual(
            td.element_spec[1], tf.TensorSpec(shape=None, dtype=tf.int64, name=None)
        )

        batch_d = next(iter(td))
        self.assertAllEqual(batch[0], batch_d[0])
        self.assertAllEqual(batch[1], batch_d[1])

    def test_serialize_deserialize_shape1(self):
        td = self.td
        batch = next(iter(td))

        td = td.map(serialize_example)
        td = td.map(make_dataset_deserialize_fn(td, set_shape=False, set_size=True))

        self.assertEqual(
            td.element_spec[0], tf.TensorSpec(shape=(None, None, None), dtype=tf.uint8)
        )
        self.assertEqual(
            td.element_spec[1], tf.TensorSpec(shape=(), dtype=tf.int64, name=None)
        )

        batch_d = next(iter(td))
        self.assertAllEqual(batch[0], batch_d[0])
        self.assertAllEqual(batch[1], batch_d[1])

    def test_serialize_deserialize_shape2(self):
        td = self.td
        batch = next(iter(td))

        td = td.map(serialize_example)
        td = td.map(make_dataset_deserialize_fn(td, set_shape=True, set_size=False))

        self.assertEqual(
            td.element_spec[0], tf.TensorSpec(shape=(28, 28, 3), dtype=tf.uint8)
        )
        self.assertEqual(
            td.element_spec[1], tf.TensorSpec(shape=(), dtype=tf.int64, name=None)
        )

        batch_d = next(iter(td))
        self.assertAllEqual(batch[0], batch_d[0])
        self.assertAllEqual(batch[1], batch_d[1])

    def test_serialize_deserialize_shape3(self):
        td = self.td
        batch = next(iter(td))

        td = td.map(serialize_example)
        td = td.map(make_dataset_deserialize_fn(td, set_shape=True, set_size=True))

        self.assertEqual(
            td.element_spec[0], tf.TensorSpec(shape=(28, 28, 3), dtype=tf.uint8)
        )
        self.assertEqual(
            td.element_spec[1], tf.TensorSpec(shape=(), dtype=tf.int64, name=None)
        )

        batch_d = next(iter(td))
        self.assertAllEqual(batch[0], batch_d[0])
        self.assertAllEqual(batch[1], batch_d[1])

    def test_serialize_deserialize_variable_shape0(self):
        td = self.td
        td = td.map(random_size)
        batch = next(iter(td))

        td = td.map(serialize_example)
        td = td.map(make_dataset_deserialize_fn(td, set_shape=False, set_size=False))

        self.assertEqual(
            td.element_spec[0], tf.TensorSpec(shape=None, dtype=tf.float32)
        )
        self.assertEqual(
            td.element_spec[1], tf.TensorSpec(shape=None, dtype=tf.int64, name=None)
        )

        batch_d = next(iter(td))
        self.assertAllEqual(batch[0], batch_d[0])
        self.assertAllEqual(batch[1], batch_d[1])

    def test_serialize_deserialize_variable_shape1(self):
        td = self.td
        td = td.map(random_size)
        batch = next(iter(td))

        td = td.map(serialize_example)
        td = td.map(make_dataset_deserialize_fn(td, set_shape=False, set_size=True))

        self.assertEqual(
            td.element_spec[0],
            tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        )
        self.assertEqual(
            td.element_spec[1], tf.TensorSpec(shape=(), dtype=tf.int64, name=None)
        )

        batch_d = next(iter(td))
        self.assertAllEqual(batch[0], batch_d[0])
        self.assertAllEqual(batch[1], batch_d[1])

    def test_serialize_deserialize_variable_shape2(self):
        td = self.td

        s0 = tf.cast(
            tf.random.uniform([16, 16, 3], maxval=255, dtype=tf.int32), tf.uint8
        )
        s1 = tf.random.uniform((), maxval=255, dtype=tf.int64)
        td_v = tf.data.Dataset.from_tensors((s0, s1))
        td = td_v.concatenate(td)

        td = td.map(serialize_example)
        td = td.map(make_dataset_deserialize_fn(td, set_shape=True, set_size=False))

        it = iter(td)
        x, y = next(it)
        with pytest.raises(InvalidArgumentError):
            x, y = next(it)
