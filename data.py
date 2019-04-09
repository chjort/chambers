import os
import tensorflow as tf
from chambers import augmentations as aug


# TODO: Figure out how to optimize data pipeline when doing random crops. Only at around 50% GPU utilization.
def load_train_test_val(path):
    sets = {"train": [], "test": [], "val": []}
    for set_ in sets.keys():
        images, masks = [], []
        for file in os.listdir(os.path.join(path, set_)):
            image = os.path.join(path, *[set_, file])
            mask = os.path.join(path, *[set_ + "_labels", file])
            images.append(image)
            masks.append(mask)
        sets[set_] = [images, masks]
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


def augmentation(img, mask):
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

        self._train = Dataset(x=sets["train"][0],
                              y=sets["train"][1],
                              class_labels=self._class_labels,
                              shuffle=True,
                              augment=True
                              )

        self._test = Dataset(x=sets["test"][0],
                             y=sets["test"][1],
                             class_labels=self._class_labels,
                             shuffle=False,
                             augment=False
                             )

        self._val = Dataset(x=sets["val"][0],
                            y=sets["val"][1],
                            class_labels=self._class_labels,
                            shuffle=False,
                            augment=False
                            )

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


class Dataset(object):
    def __init__(self, x, y, class_labels, shuffle=True, augment=False):
        """
        Creates a tf.data.dataset object

        :param x: array of file paths for x samples
        :param y: array of file paths for y samples
        """
        self._x = x
        self._y = y
        self._n = len(self._x)
        self._class_labels = class_labels

        self._tf_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            self._tf_dataset = self._tf_dataset.shuffle(buffer_size=self._n, seed=42)
        if augment:
            self._tf_dataset = self._tf_dataset.map(self._preprocess_aug_fn, num_parallel_calls=4)
        else:
            self._tf_dataset = self._tf_dataset.map(self._preprocess_fn, num_parallel_calls=4)

        self._batched_dataset = None

    @property
    def dataset(self):
        if self._batched_dataset is not None:
            return self._batched_dataset
        else:
            return self._tf_dataset

    @property
    def n(self):
        return self._n

    def _preprocess_fn(self, img_path, mask_path):
        img, mask = read_sample(img_path, mask_path)
        mask = rgb_to_onehot(mask, self._class_labels)
        img = tf.cast(img / 255, dtype=tf.float32)
        return img, mask

    def _preprocess_aug_fn(self, img_path, mask_path):
        img, mask = read_sample(img_path, mask_path)
        img, mask = augmentation(img, mask)
        mask = rgb_to_onehot(mask, self._class_labels)
        img = tf.cast(img / 255, dtype=tf.float32)
        return img, mask

    def set_batch(self, batch_size):
        n_prefetch = 1
        if batch_size == 1:
            n_prefetch = 5

        self._batched_dataset = self._tf_dataset.batch(batch_size).prefetch(n_prefetch).repeat()
