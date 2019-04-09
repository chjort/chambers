import os
import tensorflow as tf
import numpy as np
from chambers import augmentations as aug


# TODO: Figure out how to optimize data pipeline when doing random crops. Only at around 50% GPU utilization.
def load_train_test_val(path):
    sets = {"train": [], "test": [], "val": []}
    for set_ in sets.keys():
        images, masks = [], []
        for file in os.listdir(os.path.join(path, set_)):
            image = os.path.join(path, *[set_, file])
            mask = os.path.join(path, *[set_ + "_labels", file])
            # Todo: make it return [(img,mask), (img,mask)] instead of [[img,...,], [mask,...,]]
            images.append(image)
            masks.append(mask)
        sets[set_] = [images, masks]
        # Todo: Shuffle
    return sets


def rgb_to_onehot(img, class_labels):
    onehot = []
    for label in class_labels:
        binary_mask = tf.reduce_all(tf.equal(img, label), axis=-1)
        onehot.append(binary_mask)
    onehot = tf.stack(onehot, axis=-1)
    onehot = tf.cast(onehot, dtype=tf.float32)
    return onehot


def onehot_to_rgb(onehot, class_labels):
    tf_class_labels = tf.constant(class_labels, dtype=tf.uint8)
    class_indices = tf.argmax(onehot, axis=-1)
    class_indices = tf.reshape(class_indices, [-1])
    rgb_image = tf.gather(tf_class_labels, class_indices)
    rgb_image = tf.reshape(rgb_image, [onehot.shape[0], onehot.shape[1], 3])
    return rgb_image


def augment(img, mask):
    img, mask = aug.random_color(img, mask, prob=0.5)
    img, mask = aug.random_flip_horizontal(img, mask)
    img, mask = aug.random_flip_vertical(img, mask)
    img, mask = aug.random_transpose(img, mask)
    img, mask = aug.random_rot90(img, mask)
    img, mask = aug.random_rotate(img, mask, angle=180)
    return img, mask


def read_sample(img_path, mask_path):
    img_string = tf.read_file(img_path)
    mask_string = tf.read_file(mask_path)
    img = tf.image.decode_png(img_string, channels=3)
    mask = tf.image.decode_png(mask_string, channels=3)
    img = tf.cast(img, tf.float32)
    mask = tf.cast(mask, tf.float32)
    return img, mask


class Datasets(object):
    def __init__(self, data_folder):
        self._data_folder = os.path.abspath(data_folder)
        with open(os.path.join(self._data_folder, "class_dict.csv")) as f:
            data = f.read().split("\n")
            self._class_names = [class_.split(",")[0] for class_ in data[1:-1]]
            self._class_labels = [[int(rgb) for rgb in class_.split(",")[1:]] for class_ in data[1:-1]]

        sets = load_train_test_val(self._data_folder)

        tf_train = tf.data.Dataset.from_tensor_slices((sets["train"][0], sets["train"][1]))
        tf_test = tf.data.Dataset.from_tensor_slices((sets["test"][0], sets["test"][1]))
        tf_val = tf.data.Dataset.from_tensor_slices((sets["val"][0], sets["val"][1]))

        # TODO: SHUFFLE HERE INSTEAD

        tf_train = tf_train.map(self.preprocess_train, num_parallel_calls=8)
        tf_test = tf_test.map(self.preprocess_test, num_parallel_calls=8)
        tf_val = tf_val.map(self.preprocess_test, num_parallel_calls=8)

        self._train = Dataset(tf_train, len(sets["train"][0]))
        self._test = Dataset(tf_test, len(sets["test"][0]))
        self._val = Dataset(tf_val, len(sets["val"][0]))

    @property
    def data_folder(self):
        return self._data_folder

    @property
    def class_names(self):
        return self._class_names

    @property
    def class_labels(self):
        return self._class_labels

    @property
    def train(self):
        return self._train

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    def preprocess_train(self, img_path, mask_path):
        img, mask = read_sample(img_path, mask_path)
        #img, mask = aug.random_crop(img, mask, crop_shape=(256, 256))
        img, mask = augment(img, mask)
        mask = rgb_to_onehot(mask, self.class_labels)

        img = tf.cast(img / 255, dtype=tf.float32)
        return img, mask

    def preprocess_test(self, img_path, mask_path):
        img, mask = read_sample(img_path, mask_path)
        #img, mask = aug.random_crop(img, mask, crop_shape=(256, 256))
        mask = rgb_to_onehot(mask, self.class_labels)

        img = tf.cast(img / 255, dtype=tf.float32)
        return img, mask


class Dataset(object):
    def __init__(self, dataset, length=None):
        self._dataset = dataset
        self._batched_dataset = None
        self._iter = None
        self._get_batch = None

        if length is not None:
            self._length = length
        else:
            self._length = len(self._dataset)

    @property
    def dataset(self):
        if self._batched_dataset is not None:
            return self._batched_dataset
        else:
            return self._dataset

    def next_batch(self):
        assert self._get_batch is not None, "Batch size not set. Use 'set_batch' to specify batch size."
        return self._get_batch

    @property
    def n(self):
        return self._length

    def set_batch(self, batch_size, shuffle=True):
        self._batched_dataset = self._dataset.batch(batch_size).prefetch(5)  # .prefetch(batch_size)
        if shuffle:
            self._batched_dataset = self._batched_dataset.shuffle(self._length, seed=42).repeat()
        else:
            self._batched_dataset = self._batched_dataset.repeat()
        self._iter = self._batched_dataset.make_one_shot_iterator()
        self._get_batch = self._iter.get_next()
