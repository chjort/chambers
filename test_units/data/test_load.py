import os

import tensorflow as tf

from chambers.data.load import match_nested_set, match_img_files, read_and_decode_image

nested_data_path = "test_units/sample_data/mnist/train"
img_folder = os.path.join(nested_data_path, "1")
img_file = os.path.join(img_folder, "3.png")


def test_read_nested_set():
    files = match_nested_set(nested_data_path)
    assert len(files) == 10


class TestChambersDataRead(tf.test.TestCase):
    def test_read_img_files_shape(self):
        files = match_img_files(img_folder)
        self.assertAllEqual(tf.shape(files), (3,))

    def test_read_and_decode_image(self):
        img1 = read_and_decode_image(img_file, channels=1)
        self.assertAllEqual(tf.shape(img1), (28, 28, 1))

        img3 = read_and_decode_image(img_file, channels=3)
        self.assertAllEqual(tf.shape(img3), (28, 28, 3))
