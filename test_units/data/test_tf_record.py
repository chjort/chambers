import tensorflow as tf

from chambers.data.dataset import InterleaveImageDataset
from chambers.data.read import read_nested_set
from chambers.data.tf_record import serialize_example, make_dataset_deserialize_fn


class TestTFRecord(tf.test.TestCase):
    nested_data_path = "test_units/sample_data/mnist/train"
    class_dirs = sorted(read_nested_set(nested_data_path))
    labels = list(range(len(class_dirs)))

    td = InterleaveImageDataset(
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
        for x, xd in zip(batch, batch_d):
            self.assertAllEqual(x, xd)

    def test_serialize_deserialize1(self):
        td = self.td
        td = td.map(lambda x, y: (x, tf.cast(x, tf.float32), y))
        batch = next(iter(td))

        td = td.map(serialize_example)
        td = td.map(make_dataset_deserialize_fn(td))

        batch_d = next(iter(td))
        for x, xd in zip(batch, batch_d):
            self.assertAllEqual(x, xd)

    def test_serialize_deserialize2(self):
        td = self.td
        td = td.map(lambda x, y: x)
        batch = next(iter(td))

        td = td.map(serialize_example)
        td = td.map(make_dataset_deserialize_fn(td))

        batch_d = next(iter(td))
        self.assertAllEqual(batch, batch_d)