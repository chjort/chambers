import os
import tensorflow as tf
from chambers.data.tfrecord import read_feature


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


def parse_class_dict_csv(file):
    with open(file) as f:
        data = f.read().split("\n")
        class_names = [class_.split(",")[0] for class_ in data[1:-1]]
        class_labels = [[int(rgb) for rgb in class_.split(",")[1:]] for class_ in data[1:-1]]
    return class_names, class_labels


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


def read_sample(img_path, mask_path):
    img_string = tf.read_file(img_path)
    mask_string = tf.read_file(mask_path)
    img = tf.image.decode_png(img_string, channels=3)
    mask = tf.image.decode_png(mask_string, channels=3)
    img = tf.cast(img, tf.float32)
    mask = tf.cast(mask, tf.float32)
    return img, mask


class Datasets(object):
    def __init__(self, data_folder, augment_train=None, onehot=True):
        self._data_folder = os.path.abspath(data_folder)
        self._class_names, self._class_labels = parse_class_dict_csv(os.path.join(self._data_folder,
                                                                                  "class_dict.csv"))

        sets = load_train_test_val(self._data_folder)

        self._train = Dataset(x=sets["train"][0],
                              y=sets["train"][1],
                              class_labels=self._class_labels,
                              shuffle=True,
                              augmentation=augment_train,
                              one_hot=onehot
                              )

        self._test = Dataset(x=sets["test"][0],
                             y=sets["test"][1],
                             class_labels=self._class_labels,
                             shuffle=False,
                             augmentation=None,
                             one_hot=onehot
                             )

        self._val = Dataset(x=sets["val"][0],
                            y=sets["val"][1],
                            class_labels=self._class_labels,
                            shuffle=False,
                            augmentation=None,
                            one_hot=onehot
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
    def __init__(self, x, y, class_labels, shuffle=True, augmentation=None, one_hot=True):
        """
        Creates a tf.data.dataset object from image files

        :param x: array of file paths for x samples
        :param y: array of file paths for y samples
        """
        self._x = x
        self._y = y
        self._n = len(self._x)
        self._class_labels = class_labels
        self._shuffle = shuffle
        self._augmentation = augmentation
        self.one_hot = one_hot

        self._tf_dataset = self._make_pipeline()

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

    def _make_pipeline(self):
        tf_data = tf.data.Dataset.from_tensor_slices((self._x, self._y))
        if self._shuffle:
            tf_data = tf_data.shuffle(buffer_size=self._n, seed=42)
        tf_data = tf_data.map(self._preprocess, num_parallel_calls=4)

        return tf_data

    def _preprocess(self, img_path, mask_path):
        img, mask = read_sample(img_path, mask_path)
        if self._augmentation is not None:
            img, mask = self._augmentation(img, mask)
        if self.one_hot:
            mask = rgb_to_onehot(mask, self._class_labels)
        img = tf.cast(img / 255, dtype=tf.float32)
        return img, mask

    def set_batch(self, batch_size):
        n_prefetch = 1
        if batch_size == 1:
            n_prefetch = 5

        self._batched_dataset = self._tf_dataset.batch(batch_size).prefetch(n_prefetch).repeat()


class TFRecordDatasets(object):
    def __init__(self, data_folder, augment_train=True, one_hot=True):
        self._data_folder = os.path.abspath(data_folder)
        self._class_names, self._class_labels = parse_class_dict_csv(os.path.join(self._data_folder,
                                                                                  "class_dict.csv"))

        self._train = TFRecordDataset(filelist=[os.path.join(self._data_folder, "train.tfrec")],
                                      class_labels=self._class_labels,
                                      shuffle=True,
                                      augmentation=augment_train,
                                      one_hot=one_hot
                                      )

        self._test = TFRecordDataset(filelist=[os.path.join(self._data_folder, "test.tfrec")],
                                     class_labels=self._class_labels,
                                     shuffle=False,
                                     augmentation=None,
                                     one_hot=one_hot
                                     )

        self._val = TFRecordDataset(filelist=[os.path.join(self._data_folder, "val.tfrec")],
                                    class_labels=self._class_labels,
                                    shuffle=False,
                                    augmentation=None,
                                    one_hot=one_hot
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


class TFRecordDataset(object):
    def __init__(self, filelist, class_labels, shuffle=True, augmentation=None, one_hot=True):
        """
        Creates a tf.data.dataset object from a list of TFRecord files

        :param filelist: array of TFRecord files
        """
        self._file_list = filelist
        self._n = self._count_examples()
        self._class_labels = class_labels
        self._shuffle = shuffle
        self._augmentation = augmentation
        self.one_hot = one_hot

        self._tf_dataset = self._make_pipeline()

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

    def _count_examples(self):
        n = 0
        for file in self._file_list:
            for _ in tf.python_io.tf_record_iterator(file):
                n += 1
        return n

    def _make_pipeline(self):
        tf_data = tf.data.TFRecordDataset(self._file_list)
        if self._shuffle:
            tf_data = tf_data.shuffle(buffer_size=self._n, seed=42)
        # if self._augment:
        #     tf_data = tf_data.map(self._preprocess_aug, num_parallel_calls=4)
        # else:
        tf_data = tf_data.map(self._preprocess, num_parallel_calls=4)

        return tf_data

    def _preprocess(self, example):
        feature = read_feature(example)
        img = tf.image.decode_png(feature["image"], channels=3)
        mask = tf.image.decode_png(feature["label"], channels=3)
        img = tf.cast(img, tf.float32)
        mask = tf.cast(mask, tf.float32)

        if self._augmentation is not None:
            img, mask = self._augmentation(img, mask)
        if self.one_hot:
            mask = rgb_to_onehot(mask, self._class_labels)
        img = tf.cast(img / 255, dtype=tf.float32)
        return img, mask

    # def _preprocess_aug(self, example):
    #     feature = read_feature(example)
    #
    #     img = tf.image.decode_png(feature["image"], channels=3)
    #     mask = tf.image.decode_png(feature["label"], channels=3)
    #     img = tf.cast(img, tf.float32)
    #     mask = tf.cast(mask, tf.float32)
    #
    #     img, mask = augmentation(img, mask)
    #     if self.onehot:
    #         mask = rgb_to_onehot(mask, self._class_labels)
    #     img = tf.cast(img / 255, dtype=tf.float32)
    #     return img, mask

    def set_batch(self, batch_size):
        n_prefetch = 1
        if batch_size == 1:
            n_prefetch = 5

        self._batched_dataset = self._tf_dataset.batch(batch_size).prefetch(n_prefetch).repeat()
